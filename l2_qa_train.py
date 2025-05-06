import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from tqdm import tqdm
import copy

# 定义参数
parser = argparse.ArgumentParser(description="Lora训练并保存完整模型")
parser.add_argument("--model_path", type=str, default="/data/yl7622/speechLM/vicuna-7b-v1.5")
parser.add_argument("--qa_csv", type=str, default="processed_qa_data.csv")
parser.add_argument("--output_dir", type=str, default="l2_qa_lora_checkpoints")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# QA数据集
class QADataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载了 {len(self.df)} 条问答数据")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        answer = row['answer']
        
        # 组合问题和答案
        text = f"Question: {question}\nAnswer: {answer}"
        
        # 编码文本
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def save_merged_model(model, tokenizer, base_model, save_directory):
    """保存合并后的模型并修复生成配置"""
    # 创建一个新的模型实例用于合并，避免修改原始模型
    print("正在合并LoRA权重...")
    
    # 保存当前LoRA模型的状态
    temp_lora_path = os.path.join(args.output_dir, "temp_lora")
    os.makedirs(temp_lora_path, exist_ok=True)
    model.save_pretrained(temp_lora_path)
    
    # 加载一个新的基础模型
    merged_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 加载LoRA权重到新模型
    merged_model = PeftModel.from_pretrained(merged_model, temp_lora_path)
    
    # 合并LoRA权重
    merged_model = merged_model.merge_and_unload()
    
    # 创建一个一致的生成配置
    generation_config = GenerationConfig(
        temperature=0.9,
        top_p=0.6,
        do_sample=True,  # 设置为True以匹配temperature和top_p
        max_length=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 设置模型的生成配置
    merged_model.generation_config = generation_config
    
    # 保存模型和tokenizer
    merged_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # 单独保存生成配置
    generation_config.save_pretrained(save_directory)
    
    print(f"保存完整模型到 {save_directory}")

def train():
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    try:
        # 尝试直接使用LlamaTokenizer，因为Vicuna基于LLaMA
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    except:
        # 如果失败，尝试使用AutoTokenizer但禁用fast版本
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 设置LoRA配置
    print("配置LoRA参数...")
    lora_config = LoraConfig(
        r=8,  # LoRA的秩
        lora_alpha=16,  # LoRA的alpha参数
        lora_dropout=0.1,  # Dropout概率
        target_modules=["q_proj", "v_proj"],  # 目标模块
        bias="none",  # 不训练偏置项
        task_type="CAUSAL_LM"  # 任务类型
    )
    
    # 应用LoRA配置
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数
    
    # 加载数据集
    print("加载QA数据集...")
    qa_dataset = QADataset(args.qa_csv, tokenizer, args.max_length)
    
    # 准备数据加载器
    qa_loader = DataLoader(
        qa_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    total_steps = len(qa_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(qa_loader), total=len(qa_loader))
        
        for step, batch in progress_bar:
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")
            progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / len(qa_loader)
        print(f"Epoch {epoch+1}/{args.epochs} 完成, 平均损失: {avg_loss:.4f}")
        
        # 保存LoRA检查点
        lora_checkpoint_path = os.path.join(args.output_dir, f"qa_lora_checkpoint_epoch_{epoch+1}")
        model.save_pretrained(lora_checkpoint_path)
        print(f"保存LoRA检查点到 {lora_checkpoint_path}")
        
        # 保存完整模型（合并LoRA权重）
        full_checkpoint_path = os.path.join(args.output_dir, f"qa_full_model_epoch_{epoch+1}")
        save_merged_model(model, tokenizer, base_model, full_checkpoint_path)
    
    # 保存最终LoRA模型
    final_lora_path = os.path.join(args.output_dir, "qa_lora_final")
    model.save_pretrained(final_lora_path)
    print(f"保存最终LoRA模型到 {final_lora_path}")
    
    # 保存最终完整模型（合并LoRA权重）
    final_full_path = os.path.join(args.output_dir, "qa_full_model_final")
    save_merged_model(model, tokenizer, base_model, final_full_path)
    print(f"训练完成，最终完整模型保存到 {final_full_path}")

if __name__ == "__main__":
    train()