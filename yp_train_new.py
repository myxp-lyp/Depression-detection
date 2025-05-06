import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
#torchaudio.set_audio_backend("sox_io")
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoProcessor, Wav2Vec2Model, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import librosa
from torch.utils.data import Sampler
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """
    Set seed for reproducibility in Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): Whether to enforce deterministic algorithms (for exact reproducibility).
        benchmark (bool): Whether to enable cudnn.benchmark (for faster but non-deterministic training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    if deterministic:
        torch.backends.cudnn.deterministic = True  # Enforce determinism
    if not benchmark:
        torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

# Example usage
set_seed(42)



# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ==================== 1. 读取 CSV，建立 ID -> Score 映射 ====================
csv_path = "train_split_Depression_AVEC2017.csv"  # 替换成你的 CSV 路径
df = pd.read_csv(csv_path)
id_to_score_train = dict(zip(df["Participant_ID"].astype(str), df["PHQ8_Score"]))

# ==================== 2. 处理音频文件，筛选有效文件 ====================
audio_folder = "preprocessed_audio"  # 替换成你的音频文件夹路径
train_f = [
    file for file in os.listdir(audio_folder)
    if file.endswith(".wav") and file.split("_")[0] in id_to_score_train
]
train_files = train_f

csv_path = "dev_split_Depression_AVEC2017.csv"  # 替换成你的 CSV 路径
df = pd.read_csv(csv_path)
id_to_score_dev = dict(zip(df["Participant_ID"].astype(str), df["PHQ8_Score"]))
dev_f = [
    file for file in os.listdir(audio_folder)
    if file.endswith(".wav") and file.split("_")[0] in id_to_score_dev
]
dev_files = dev_f


csv_path = "full_test_split.csv"  # 替换成你的 CSV 路径
df = pd.read_csv(csv_path)
id_to_score_test = dict(zip(df["Participant_ID"].astype(str), df["PHQ8_Score"]))
test_f = [
    file for file in os.listdir(audio_folder)
    if file.endswith(".wav") and file.split("_")[0] in id_to_score_test
]
test_files = test_f

#print(f"训练集: {len(train_files)}，验证集：{len(dev_files)}，测试集: {len(test_files)}")

# ==================== 3. 定义 Dataset 类 ====================
def pad_waveform(waveform, max_length):
    """
    对 waveform 进行 padding，使所有样本长度一致
    :param waveform: torch.Tensor (1, seq_len)
    :param max_length: 目标长度
    :return: 统一长度的 waveform
    """
    cur_length = waveform.shape[-1]
    if cur_length < max_length:
        # 计算填充值
        pad_size = max_length - cur_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))  # 右侧填充
    else:
        # 截断
        waveform = waveform[:, :max_length]
    return waveform

class GroupedSampler(Sampler):
    def __init__(self, file_list, batch_size):
        self.batch_size = batch_size
        self.groups = defaultdict(list)
        
        # 按 participant_id 分组
        for idx, file_name in enumerate(file_list):
            participant_id = file_name.split("_")[0]
            self.groups[participant_id].append(idx)

        # 将 groups 转换成一个列表
        self.grouped_indices = list(self.groups.values())
        random.shuffle(self.grouped_indices)  # 打乱 group 顺序

    def __iter__(self):
        batch = []
        for group in self.grouped_indices:
            batch.extend(group)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # 剩余的不足一个 batch 的样本
        if batch:
            yield batch

    def __len__(self):
        return len(self.grouped_indices)

def collate_fn(batch):
    encoding1, audio_features, encoding2, scores = zip(*batch)  # 分离特征和分数
    audio_features = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True, padding_value=0)
    encoding1 = torch.stack(encoding1)
    encoding2 = torch.stack(encoding2) 
    scores = torch.stack(scores)
    return encoding1, audio_features, encoding2, scores

class AudioDataset(Dataset):
    def __init__(self, audio_folder, file_list, processor, id_to_score):
        self.audio_folder = audio_folder
        self.file_list = file_list
        self.processor = processor
        self.id_to_score = id_to_score

    def __len__(self):
        return len(self.file_list)

    def load_audio_librosa(file_path, target_sr=16000):
        try:
            # 使用 librosa 加载音频，自动转换为目标采样率
            waveform, sample_rate = librosa.load(file_path, sr=target_sr)
            return waveform, sample_rate
        except Exception as e:
            print(f"⚠️ 无法加载音频 {file_path}: {e}")
            return None, None  # 解析失败返回 None

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        participant_id = file_name.split("_")[0]
        score = self.id_to_score[participant_id]
        # 读取音频
        file_path = os.path.join(self.audio_folder, file_name)
        waveform, sample_rate = torchaudio.load(file_path)
        max_length = 160000
        if waveform is not None:
            waveform = torch.as_tensor(waveform)#.unsqueeze(0)  # 转成 Tensor 并加上 batch 维度
            waveform = pad_waveform(waveform, max_length)
        # 预处理音频
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        audio_features = inputs.input_values.squeeze(0)  # [seq_len]
        encoding1 = tokenizer(
            prompts1,
            truncation=False,
            return_tensors="pt"
        )['input_ids']
        encoding2 = tokenizer(
                prompts2,
                truncation=False,
                return_tensors="pt"
            )['input_ids']
        
        score = tokenizer(str(score), padding="max_length", max_length=5, return_tensors="pt")['input_ids']

        return encoding1.squeeze(0), audio_features.squeeze(0), encoding2.squeeze(0), score.squeeze(0)#, dtype=torch.float32)

# ==================== 4. 创建训练集 & 测试集 ====================
def prepare_dataloaders(rank, world_size, train_files, dev_files, test_files, processor, id_to_score, batch_size=8):
    train_dataset = AudioDataset(audio_folder, train_files, processor, id_to_score[0])
    dev_dataset = AudioDataset(audio_folder, dev_files, processor, id_to_score[1])
    test_dataset = AudioDataset(audio_folder, test_files, processor, id_to_score[2])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=17, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=dev_sampler, num_workers=17, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=17, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader

# ==================== 5. 初始化 Wav2Vec2 & LLaMA ====================
def prepare_models(rank):
    wav2vec_model = Wav2Vec2Model.from_pretrained("/data/yl7622/wav2vec2").to(rank)
    for param in wav2vec_model.parameters():
        param.requires_grad = False  # 冻结 Wav2Vec2
    basemodel = AutoModelForCausalLM.from_pretrained("/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf").to(rank)
 
    # LoRA 适配器
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]
    )
    llama_model = get_peft_model(basemodel, lora_config).to(rank)
    llama_model = DDP(llama_model, device_ids=[rank])

    return wav2vec_model, llama_model

tokenizer = AutoTokenizer.from_pretrained("/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf")
prompts1 = 'Audio: '
prompts2 = "\nPHQ Score:"
tokenizer.pad_token = tokenizer.eos_token


def prepare_models_use_non_DDPllama():
    wav2vec_model = Wav2Vec2Model.from_pretrained("/data/yl7622/wav2vec2")
    for param in wav2vec_model.parameters():
        param.requires_grad = False  # 冻结 Wav2Vec2 
    # LoRA 适配器
    basemodel = AutoModelForCausalLM.from_pretrained("/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf")



    llama_model = basemodel#PeftModel.from_pretrained(basemodel, '/data/yl7622/emotion_detection/qa_lora_final')
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]
    )
    llama_model = get_peft_model(llama_model, lora_config)
    llama_model.train()
    return wav2vec_model, llama_model

# ==================== 6. 定义 Projector (用于对齐特征) ====================
class AudioToTextProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)  # 线性投影
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.proj(x))

class ValueProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)  # 线性投影

    def forward(self, x):
        return self.proj(x)
    
def prepare_projectors(rank, wav2vec_model, llama_model):

    projector = AudioToTextProjector(wav2vec_model.config.hidden_size, llama_model.module.config.hidden_size).to(rank)
    projector2 = ValueProjector(32000, 1).to(rank)
    projector = DDP(projector, device_ids=[rank])
    projector2 = DDP(projector2, device_ids=[rank])
    return projector, projector2



# ==================== 7. 训练代码 ====================
def train_epoch(rank, wav2vec_model, llama_model, projector, projector2, dataloader, optimizer, loss_fn):
    wav2vec_model.eval()
    llama_model.train()
    projector.train()
    projector2.train()
    total_loss = 0

    for encoding1, audio_features, encoding2, scores in dataloader:
        encoding1, audio_features, encoding2, scores = encoding1.to(rank),audio_features.to(rank), encoding2.to(rank),scores.to(rank)
        optimizer.zero_grad()

        # Wav2Vec2 提取特征
        audio_embeds = wav2vec_model(audio_features).last_hidden_state  # [batch, seq_len, hidden_size]
        # 投影到 LLaMA 维度
        audio_embeds = projector(audio_embeds.mean(dim=1)).unsqueeze(1)  # [batch, hidden_size]


        

        encoding1 = llama_model.module.get_input_embeddings()(encoding1)
        encoding2 = llama_model.module.get_input_embeddings()(encoding2)
        scores = llama_model.module.get_input_embeddings()(scores)

        inputs = torch.cat([encoding1, audio_embeds, encoding2, scores], dim=1)
        labels = inputs.clone()
        #TODO: need debug
        # 送入 LLaMA
        outputs = llama_model(inputs_embeds=inputs, labels=labels)  
            
        #outputs = llama_model(inputs_embeds=audio_embeds).logits  #
        #  [batch, seq_len, vocab_size]

        # 取最后一层的均值作为回归值
        #predictions = projector2(outputs[:, -1])#outputs[:, -1, 0]  # [batch]
        #loss = loss_fn(predictions.view(-1), scores)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        #print(loss.item())

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(rank, wav2vec_model, llama_model, projector, projector2, dataloader, loss_fn):
    wav2vec_model.eval()
    llama_model.eval()
    projector.eval()
    projector2.eval()
    total_loss = 0

    with torch.no_grad():
        for audio_features, scores in dataloader:
            audio_features, scores = audio_features.to(rank), scores.to(rank)

            # Wav2Vec2 提取特征
            audio_embeds = wav2vec_model(audio_features).last_hidden_state  # [batch, seq_len, hidden_size]

            # 投影到 LLaMA 维度
            audio_embeds = projector(audio_embeds.mean(dim=1)).unsqueeze(1)  # [batch, hidden_size]

            # 送入 LLaMA
            outputs = llama_model(inputs_embeds=audio_embeds).logits  # [batch, seq_len, vocab_size]

            # 取最后一层的均值作为回归值
            predictions = projector2(outputs[:, -1])#outputs[:, -1, 0]  # [batch]
            loss = loss_fn(predictions.view(-1), scores)
            total_loss += loss.item()
            #print(loss.item())

    return total_loss / len(dataloader)

def generate_evaluate(rank, wav2vec_model, llama_model, projector, projector2, dataloader, loss_fn):
    wav2vec_model.eval()
    llama_model.eval()
    projector.eval()
    projector2.eval()
    total_loss = 0

    with torch.no_grad():
        for audio_features, scores in dataloader:
            audio_features, scores = audio_features.to(rank), scores.to(rank)

            # Wav2Vec2 提取特征
            audio_embeds = wav2vec_model(audio_features).last_hidden_state  # [batch, seq_len, hidden_size]

            # 投影到 LLaMA 维度
            audio_embeds = projector(audio_embeds.mean(dim=1)).unsqueeze(1)  # [batch, hidden_size]
            
            #TODO: need debug
            prompt_encodings = encoding1 + audio_embeds + encoding2
            # 送入 LLaMA
            #outputs = llama_model(inputs_embeds=audio_embeds).logits  # [batch, seq_len, vocab_size]
            generated = llama_model.generate(
                            **prompt_encodings,
                            max_new_tokens=10,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0
            )

            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                        
                        # Extract predicted PHQ score
            try:
                phq_part = generated_text.split("PHQ Score:")[1].strip()
                predicted_score = float(''.join([c for c in phq_part if c.isdigit() or c == '.'][:5]))
            except:
                predicted_score = 10.0
            # 取最后一层的均值作为回归值
            #predictions = projector2(outputs[:, -1])#outputs[:, -1, 0]  # [batch]
            #loss = loss_fn(predictions.view(-1), scores)
            loss = loss_fn(predicted_score, scores)
            total_loss += loss.item()
            print(loss.item())

    return total_loss / len(dataloader)

def load_checkpoint(rank, projector, projector2, llama_model, optimizer, save_dir):
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        print("⚠️ 没有找到已有的检查点，训练将从头开始")
        return 0, projector, projector2, llama_model, optimizer # 返回起始 epoch = 0
    
    # 找到最新的 checkpoint
    checkpoints.sort(reverse=True)
    latest_checkpoint = os.path.join(save_dir, checkpoints[0])
    checkpoint = torch.load(latest_checkpoint)
    epoch = checkpoint["epoch"]
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector2.load_state_dict(checkpoint["projector2_state_dict"])
    llama_model = PeftModel.from_pretrained(llama_model, f"lora_checkpoint_{epoch}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"🔄 加载权重: {latest_checkpoint}")
    return checkpoint["epoch"] + 1, projector, projector2, llama_model, optimizer   # 从下一个 epoch 开始训练

def save_model(rank, epoch, projector, projector2, llama_model, optimizer, save_dir):
    if rank == 0:  # Only save on the main process
        save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        # full_model = llama_model.module.merge_and_unload()
        llama_model.module.save_pretrained(f"lora_{epoch}")

        # Save the full model
        # full_model.save_pretrained(f"llama_checkpoint_{epoch}")        
        torch.save({
                "epoch": epoch,
                "projector_state_dict": projector.state_dict(),
                "projector2_state_dict": projector2.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
        print(f"✅ 模型已保存: {save_path}")
    torch.distributed.barrier()

# ==================== 8. 训练 & 评估 ====================
def main(rank, world_size):
    wav2vec_model, llama_model = prepare_models_use_non_DDPllama()
    

    setup(rank, world_size)

    # uodate the model to DDP. This is for non_DDPllama.
    wav2vec_model.to(rank)
    llama_model.to(rank)
    llama_model = DDP(llama_model, device_ids=[rank])

    save_dir = '/data/yl7622/emotion_detection/checkpoints'
    processor = AutoProcessor.from_pretrained("/data/yl7622/wav2vec2")
    id_to_score = [id_to_score_train, id_to_score_dev, id_to_score_test]
    train_loader, dev_loader, test_loader = prepare_dataloaders(rank, world_size, train_files, dev_files,test_files, processor, id_to_score)
    # wav2vec_model, llama_model = prepare_models(rank)
    projector, projector2 = prepare_projectors(rank, wav2vec_model, llama_model)

    loss_fn = nn.L1Loss()#nn.MSELoss()
    optimizer = torch.optim.Adam(list(projector.parameters()) + list(projector2.parameters()) + list(llama_model.parameters()), lr=1e-4)

    num_epochs = 6
    start_epoch, projector, projector2, llama_model, optimizer = load_checkpoint(rank, projector, projector2, llama_model, optimizer, save_dir)
    for epoch in range(start_epoch, num_epochs):  # 训练 5 轮
        print(f"Epoch: {epoch}")
        train_loss = train_epoch(rank, wav2vec_model, llama_model, projector, projector2, train_loader, optimizer, loss_fn)
        dev_loss = evaluate(rank, wav2vec_model, llama_model, projector, projector2, dev_loader, loss_fn)

        #test_loss = evaluate(rank, wav2vec_model, llama_model, projector, projector2, test_loader, loss_fn)

        #print(test_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}")
        save_model(rank, epoch, projector, projector2, llama_model, optimizer, save_dir)

    cleanup()

if __name__ == "__main__":

    
    world_size = 2  # Number of GPUs
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)