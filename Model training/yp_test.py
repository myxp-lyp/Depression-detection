import os
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("sox_io")
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoProcessor, Wav2Vec2Model, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,PeftModel
import librosa
from torch.utils.data import Sampler
from collections import defaultdict
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



# ==================== 2. 处理音频文件，筛选有效文件 ====================
audio_folder = "preprocessed_audio"  # 替换成你的音频文件夹路径
#csv_path = "/data/yl7622/emotion_detection/dev_split_Depression_AVEC2017.csv"
csv_path = "full_test_split.csv"  # 替换成你的 CSV 路径
df = pd.read_csv(csv_path)
id_to_score = dict(zip(df["Participant_ID"].astype(str), df["PHQ8_Score"]))
test_f = [
    file for file in os.listdir(audio_folder)
    if file.endswith(".wav") and file.split("_")[0] in id_to_score
]
test_files = test_f
pred_with_id = {key: torch.zeros([0]) for key in id_to_score}
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
        return audio_features.squeeze(0), torch.tensor(score, dtype=torch.float32), file_name

# ==================== 4. 创建训练集 & 测试集 ====================

# ==================== 4. 创建训练集 & 测试集 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("/data/yl7622/wav2vec2")

test_dataset = AudioDataset(audio_folder, test_files, processor, id_to_score)

test_loader = DataLoader(test_dataset, batch_size = 1)

# ==================== 5. 初始化 Wav2Vec2 & LLaMA ====================
wav2vec_model = Wav2Vec2Model.from_pretrained("/data/yl7622/wav2vec2").to(device)
for param in wav2vec_model.parameters():
    param.requires_grad = False  # 冻结 Wav2Vec2


basemodel = AutoModelForCausalLM.from_pretrained("/data/yl7622/emotion_detection/models--meta-llama--Llama-2-7b-hf").to(device)

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]
)
llama_model = get_peft_model(basemodel, lora_config).to(device)


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
        self.proj = nn.Linear(input_dim, 1)  # 线性投影

    def forward(self, x):
        return self.proj(x)
    
projector = AudioToTextProjector(wav2vec_model.config.hidden_size, llama_model.config.hidden_size).to(device)
projector2 = ValueProjector(32000, 1).to(device)

# ==================== 7. 训练代码 ====================
loss_fn = nn.L1Loss()#nn.MSELoss()

def evaluate(model, projector, projector2, dataloader, loss_fn):
    model.eval()
    projector.eval()

    with torch.no_grad():
        for audio_features, scores, file_name in dataloader:
            audio_features, scores = audio_features.to(device), scores.to(device)

            # Wav2Vec2 提取特征
            audio_embeds = wav2vec_model(audio_features).last_hidden_state  # [batch, seq_len, hidden_size]

            # 投影到 LLaMA 维度
            audio_embeds = projector(audio_embeds.mean(dim=1)).unsqueeze(1)  # [batch, hidden_size]

            # 送入 LLaMA
            outputs = llama_model(inputs_embeds=audio_embeds).logits  # [batch, seq_len, vocab_size]

            # 取最后一层的均值作为回归值
            predictions = projector2(outputs[:, -1])#outputs[:, -1, 0]  # [batch]
            pred_with_id[file_name[0][:3]] = torch.cat((pred_with_id[file_name[0][:3]],predictions[0].cpu()))
            #loss = loss_fn(predictions, scores)
            #total_loss += loss.item()
            #print(loss.item())
    total_loss = 0
    num = 0
    for key, value in pred_with_id.items():
        num += 1
        score = id_to_score[key]
        pred = torch.mean(value)
        mae = torch.abs(score - pred)
        total_loss += mae
    return total_loss / num

def change_ddp_to_noddp(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    # Load the updated state_dict into the model
    return new_state_dict
def load_checkpoint(projector, projector2, llama_model, save_dir='checkpoints'):
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")]
    
    # 找到最新的 checkpoint
    checkpoints.sort(reverse=True)
    latest_checkpoint = os.path.join(save_dir, checkpoints[-1])
    checkpoint = torch.load(latest_checkpoint)
    epoch = checkpoint["epoch"]
    checkpoint = torch.load(latest_checkpoint)

    projector.load_state_dict(change_ddp_to_noddp(checkpoint["projector_state_dict"]))
    projector2.load_state_dict(change_ddp_to_noddp(checkpoint["projector2_state_dict"]))
    llama_model_2 = PeftModel.from_pretrained(llama_model, f"lora_{epoch}")
    
    print(f"🔄 加载权重: {latest_checkpoint}")
    return projector, projector2, llama_model   # 从下一个 epoch 开始训练


# ==================== 8. 训练 & 评估 ====================
projector, projector2, llama_model = load_checkpoint(projector, projector2, llama_model)
test_loss = evaluate(llama_model, projector, projector2, test_loader, loss_fn)
print(test_loss)
