import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from tqdm import tqdm

# 定义参数
parser = argparse.ArgumentParser(description="Lora")
parser.add_argument("--model_path", type=str, default="/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf")
parser.add_argument("--qa_csv", type=str, default="processed_qa_data.csv")
parser.add_argument("--output_dir", type=str, default="qa_lora_checkpoints")
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
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # 使用输入作为标签进行自回归训练
        }

def train():
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        
        # 保存检查点
        checkpoint_path = os.path.join(args.output_dir, f"qa_lora_checkpoint_epoch_{epoch+1}")
        model.save_pretrained(checkpoint_path)
        print(f"保存检查点到 {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "qa_lora_final")
    model.save_pretrained(final_path)
    print(f"训练完成，最终模型保存到 {final_path}")

if __name__ == "__main__":
    train()