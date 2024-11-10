import os
import zipfile
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

"""
预处理数据部分(从xxx_P.zip文件开始)
"""
def load_phq8_labels(label_files):
    phq8_dict = {}
    for file in label_files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            participant_id = str(row['Participant_ID'])
            phq8_score = row['PHQ8_Score']
            phq8_dict[participant_id] = phq8_score
    return phq8_dict

label_files = [
    '/content/drive/MyDrive/Colab Notebooks/depression_dataset/dev_split_Depression_AVEC2017.csv',
    '/content/drive/MyDrive/Colab Notebooks/depression_dataset/train_split_Depression_AVEC2017.csv'
]

phq8_labels = load_phq8_labels(label_files)

def process_data(zip_dir, phq8_labels, output_dir):
    data = []

    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('_P.zip')]

    for zip_file in tqdm(zip_files, desc='Processing zip files'):
        participant_id = zip_file.split('_')[0]
        if participant_id not in phq8_labels:
            print(f"Participant {participant_id} not found in PHQ-8 labels.")
            continue

        phq8_score = phq8_labels[participant_id]

        zip_path = os.path.join(zip_dir, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        transcript_file = os.path.join(output_dir, f"{participant_id}_TRANSCRIPT.csv")
        audio_file = os.path.join(output_dir, f"{participant_id}_AUDIO.wav")

        if not os.path.exists(transcript_file) or not os.path.exists(audio_file):
            print(f"Files for participant {participant_id} are incomplete.")
            continue

        try:
            transcript_df = pd.read_csv(transcript_file, sep=None, engine='python') 
            if 'value' not in transcript_df.columns:
                print(f"'value' column not found in {transcript_file}.")
                continue

            text_content = transcript_df['value'].astype(str).tolist()
            sentences = text_content
        except Exception as e:
            print(f"Error reading {transcript_file}: {e}")
            continue

        try:
            waveform, sample_rate = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error reading {audio_file}: {e}")
            continue

        data.append({
            'participant_id': participant_id,
            'phq8_score': phq8_score,
            'sentences': sentences,
            'audio_waveform': waveform,
            'sample_rate': sample_rate
        })

        os.remove(transcript_file)
        os.remove(audio_file)

    return data

zip_directory = '/content/drive/MyDrive/Colab Notebooks/depression_dataset' 
output_directory = './temp_unzip'
os.makedirs(output_directory, exist_ok=True)

dataset = process_data(zip_directory, phq8_labels, output_directory)

class PHQ8Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentences = item['sentences']
        audio_waveform = item['audio_waveform']
        phq8_score = item['phq8_score']
        return sentences, audio_waveform, phq8_score

def collate_fn(batch):
    sentences_batch, audio_waveforms_batch, phq8_scores_batch = zip(*batch)

    phq8_scores_batch = torch.tensor(phq8_scores_batch, dtype=torch.float32)

    return sentences_batch, audio_waveforms_batch, phq8_scores_batch

batch_size = 2  # Batch size
phq8_dataset = PHQ8Dataset(dataset)
data_loader = DataLoader(phq8_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

"""
模型部分
"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class HiBERT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(HiBERT, self).__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, encoded_sentences):
        _, hidden_states = self.rnn(encoded_sentences)
        document_representation = hidden_states[-1] 
        return document_representation

def encode_sentences(batch_of_documents):
    batch_encoded_documents = []
    for sentences in batch_of_documents:
        encoded_sentences = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = bert_model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            encoded_sentences.append(sentence_embedding)
        if len(encoded_sentences) > 0:
            encoded_sentences = torch.stack(encoded_sentences)
            batch_encoded_documents.append(encoded_sentences)
        else:
            batch_encoded_documents.append(torch.zeros(1, 768).to(device))
    batch_encoded_documents = nn.utils.rnn.pad_sequence(batch_encoded_documents, batch_first=True)
    return batch_encoded_documents

mel_spectrogram = MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80).to(device)

class Conformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Conformer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.linear(x))
        output = output.mean(dim=1)
        return output  

conformer_model = Conformer(input_dim=80, hidden_dim=256).to(device)

def process_audio(audio_waveforms):
    batch_audio_features = []
    for waveform in audio_waveforms:
        waveform = waveform.to(device)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  
        audio_features = mel_spectrogram(waveform) 
        audio_features = audio_features.transpose(1, 2) 
        batch_audio_features.append(audio_features.squeeze(0))
    batch_audio_features = nn.utils.rnn.pad_sequence(batch_audio_features, batch_first=True)
    audio_representation = conformer_model(batch_audio_features)
    return audio_representation 

def fuse_modalities(text_representation, audio_representation):
    combined_representation = torch.cat((text_representation, audio_representation), dim=-1)
    return combined_representation

class PHQ8Predictor(nn.Module):
    def __init__(self, input_dim):
        super(PHQ8Predictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        output = self.linear(x)
        return output.squeeze(-1) 

class MultiModalPHQ8Model(nn.Module):
    def __init__(self, text_model, predictor):
        super(MultiModalPHQ8Model, self).__init__()
        self.text_model = text_model
        self.predictor = predictor

    def forward(self, batch_of_sentences, audio_waveforms):
        encoded_sentences = encode_sentences(batch_of_sentences)
        text_representation = self.text_model(encoded_sentences)

        audio_representation = process_audio(audio_waveforms)

        combined_representation = fuse_modalities(text_representation, audio_representation)

        output = self.predictor(combined_representation)
        return output

text_model = HiBERT(embedding_dim=768, hidden_dim=256).to(device)
predictor = PHQ8Predictor(input_dim=512).to(device)
multi_modal_model = MultiModalPHQ8Model(text_model, predictor).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(multi_modal_model.parameters(), lr=1e-4)

num_epochs = 5  # Epochs
multi_modal_model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in data_loader:
        sentences_batch, audio_waveforms_batch, phq8_scores_batch = batch

        optimizer.zero_grad()

        outputs = multi_modal_model(sentences_batch, audio_waveforms_batch)
        targets = phq8_scores_batch.to(device)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')


"""
对特定数据进行测试
(因为没找到test split的PHQ8所以暂时没做evaluation matrix的完整测试)
"""

multi_modal_model.eval()

participant_id = '307' # ID
zip_directory = '/content/drive/MyDrive/Colab Notebooks/depression_dataset' 
output_directory = './temp_unzip'
os.makedirs(output_directory, exist_ok=True)

zip_file = os.path.join(zip_directory, f'{participant_id}_P.zip')

if not os.path.exists(zip_file):
    print(f"Zip file for participant {participant_id} not found.")
    exit()

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_directory)

transcript_file = os.path.join(output_directory, f"{participant_id}_TRANSCRIPT.csv")
audio_file = os.path.join(output_directory, f"{participant_id}_AUDIO.wav")

if not os.path.exists(transcript_file) or not os.path.exists(audio_file):
    print(f"Files for participant {participant_id} are incomplete.")
    exit()

try:
    transcript_df = pd.read_csv(transcript_file, sep=None, engine='python')
    if 'value' not in transcript_df.columns:
        print(f"'value' column not found in {transcript_file}.")
        exit()
    text_content = transcript_df['value'].astype(str).tolist()
    sentences = text_content
except Exception as e:
    print(f"Error reading {transcript_file}: {e}")
    exit()

try:
    waveform, sample_rate = torchaudio.load(audio_file)
except Exception as e:
    print(f"Error reading {audio_file}: {e}")
    exit()

os.remove(transcript_file)
os.remove(audio_file)

sentences_batch = [sentences]
audio_waveforms_batch = [waveform]

with torch.no_grad():
    outputs = multi_modal_model(sentences_batch, audio_waveforms_batch)
    predicted_phq8_score_tensor = outputs

    predicted_phq8_score = predicted_phq8_score_tensor.item()

    predicted_phq8_score = max(0, min(predicted_phq8_score, 24))

    print(f'Predicted PHQ-8 Score for participant {participant_id}: {predicted_phq8_score:.2f}')