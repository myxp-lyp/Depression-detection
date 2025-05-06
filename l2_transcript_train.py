import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define parameters
parser = argparse.ArgumentParser(description="PHQ Score Prediction with LoRA and Full Model Saving")
parser.add_argument("--model_path", type=str, default="/data/yl7622/speechLM/vicuna-7b-v1.5")
parser.add_argument("--train_csv", type=str, default="train_split_Depression_AVEC2017.csv")
parser.add_argument("--dev_csv", type=str, default="dev_split_Depression_AVEC2017.csv")
parser.add_argument("--test_csv", type=str, default="full_test_split.csv")
parser.add_argument("--transcript_csv", type=str, default="processed_transcript.csv")
parser.add_argument("--output_dir", type=str, default="phq_prediction_checkpoints")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

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
set_seed(args.seed)

# PHQ Dataset
class PHQDataset(Dataset):
    def __init__(self, transcript_df, phq_df, tokenizer, max_length=512):
        # Filter transcripts to only include participants in the PHQ dataframe
        participant_ids = phq_df['Participant_ID'].unique()
        self.transcript_df = transcript_df[transcript_df['id'].isin(participant_ids)]
        self.phq_df = phq_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create a mapping from participant ID to PHQ score
        self.phq_scores = {row['Participant_ID']: row['PHQ8_Score'] 
                          for _, row in phq_df.iterrows()}
        
        print(f"Loaded {len(self.transcript_df)} text samples for {len(participant_ids)} participants")
        
    def __len__(self):
        return len(self.transcript_df)
    
    def __getitem__(self, idx):
        row = self.transcript_df.iloc[idx]
        text = row['index']  # Transcript content
        participant_id = row['id']  # Participant ID
        phq_score = self.phq_scores[participant_id]  # Get PHQ score for this participant
        
        # Build prompt template
        prompt = f"Transcript: {text}\nPHQ Score:"
        completion = f" {phq_score}"
        
        full_text = prompt + completion
        
        # Encode text
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # Calculate prompt length
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_length = prompt_encodings["input_ids"].shape[1]
        
        # Create labels, only consider loss for completion part
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Ignore loss for prompt part
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "participant_id": participant_id
        }

def save_merged_model(model, tokenizer, save_directory):
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
        temperature=1.0,
        top_p=1.0,
        do_sample=False,  # 设置为False以匹配PHQ预测的确定性生成
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
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    transcript_df = pd.read_csv(args.transcript_csv, sep=',')
    train_phq_df = pd.read_csv(args.train_csv, sep=',')
    dev_phq_df = pd.read_csv(args.dev_csv, sep=',')
    test_phq_df = pd.read_csv(args.test_csv, sep=',')
    
    # Convert participant IDs to string format for consistency
    transcript_df['id'] = transcript_df['id'].astype(str)
    train_phq_df['Participant_ID'] = train_phq_df['Participant_ID'].astype(str)
    dev_phq_df['Participant_ID'] = dev_phq_df['Participant_ID'].astype(str)
    test_phq_df['Participant_ID'] = test_phq_df['Participant_ID'].astype(str)
    
    # Remove NaN values
    transcript_df = transcript_df.dropna(subset=['index'])
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        # 尝试直接使用LlamaTokenizer，因为Vicuna基于LLaMA
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    except:
        # 如果失败，尝试使用AutoTokenizer但禁用fast版本
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Configure LoRA
    print("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha parameter
        lora_dropout=0.1,  # Dropout probability
        target_modules=["q_proj", "v_proj"],  # Target modules for Llama
        bias="none",  # Don't train bias terms
        task_type="CAUSAL_LM"  # Causal language modeling task
    )
    
    # Apply LoRA configuration
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PHQDataset(transcript_df, train_phq_df, tokenizer, args.max_length)
    val_dataset = PHQDataset(transcript_df, dev_phq_df, tokenizer, args.max_length)
    test_dataset = PHQDataset(transcript_df, test_phq_df, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for step, batch in progress_bar:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_participant_ids = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                participant_ids = batch["participant_id"]
                
                for i in range(len(participant_ids)):
                    # Get text for current sample
                    participant_id = participant_ids[i]
                    sample_df = transcript_df[transcript_df['id'] == participant_id]
                    
                    if len(sample_df) == 0:
                        continue
                        
                    text = sample_df.iloc[0]['index']
                    
                    # Build prompt
                    prompt_text = f"Transcript: {text}\nPHQ Score:"
                    prompt_encodings = tokenizer(
                        prompt_text,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate answer
                    generated = model.generate(
                        **prompt_encodings,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0
                    )
                    
                    # Decode generated text
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Extract predicted PHQ score
                    try:
                        # Extract number from generated text
                        phq_part = generated_text.split("PHQ Score:")[1].strip()
                        # Extract first number
                        predicted_score = float(''.join([c for c in phq_part if c.isdigit() or c == '.'][:5]))
                    except:
                        # Use default value if extraction fails
                        predicted_score = 10.0
                    
                    all_preds.append(predicted_score)
                    all_participant_ids.append(participant_id)
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame({
            'Participant_ID': all_participant_ids,
            'pred_score': all_preds
        })
        
        # Calculate average predicted score by participant ID
        avg_pred = pred_df.groupby('Participant_ID')['pred_score'].mean().reset_index()
        
        # Get true PHQ scores for validation set
        true_phq = dev_phq_df[['Participant_ID', 'PHQ8_Score']]
        
        # Merge predictions with true scores
        merged_df = pd.merge(avg_pred, true_phq, on='Participant_ID')
        
        # Calculate evaluation metrics
        mse = mean_squared_error(merged_df['PHQ8_Score'], merged_df['pred_score'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(merged_df['PHQ8_Score'], merged_df['pred_score'])
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")
        print(f"Validation results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Evaluated participants: {len(merged_df)}")
        
        # Save LoRA checkpoint
        lora_checkpoint_path = os.path.join(args.output_dir, f"lora_checkpoint_epoch_{epoch+1}")
        model.save_pretrained(lora_checkpoint_path)
        print(f"Saved LoRA checkpoint to {lora_checkpoint_path}")
        
        # Save full model (merged LoRA weights)
        full_checkpoint_path = os.path.join(args.output_dir, f"full_model_epoch_{epoch+1}")
        save_merged_model(model, tokenizer, full_checkpoint_path)
        
        # Save best model (based on MAE)
        if mae < best_mae:
            best_mae = mae
            best_lora_path = os.path.join(args.output_dir, "best_lora_model")
            model.save_pretrained(best_lora_path)
            
            best_full_path = os.path.join(args.output_dir, "best_full_model")
            save_merged_model(model, tokenizer, best_full_path)
            print(f"Saved best model (MAE: {mae:.4f})")
    
    # Training complete
    print(f"Training complete, best MAE: {best_mae:.4f}")
    
    # Test on test set using best model
    print("Evaluating on test set using best model...")
    
    # Load a fresh base model
    test_base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Load the best LoRA model
    model = PeftModel.from_pretrained(test_base_model, os.path.join(args.output_dir, "best_lora_model"))
    model.eval()
    
    all_test_preds = []
    all_test_participant_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            participant_ids = batch["participant_id"]
            
            for participant_id in participant_ids:
                # Get all transcripts for this participant
                participant_transcripts = transcript_df[transcript_df['id'] == participant_id]
                
                if len(participant_transcripts) == 0:
                    continue
                    
                participant_preds = []
                
                for _, row in participant_transcripts.iterrows():
                    text = row['index']
                    
                    # Build prompt
                    prompt_text = f"Transcript: {text}\nPHQ Score:"
                    prompt_encodings = tokenizer(
                        prompt_text,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate answer
                    generated = model.generate(
                        **prompt_encodings,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0
                    )
                    
                    # Decode generated text
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Extract predicted PHQ score
                    try:
                        phq_part = generated_text.split("PHQ Score:")[1].strip()
                        predicted_score = float(''.join([c for c in phq_part if c.isdigit() or c == '.'][:5]))
                    except:
                        predicted_score = 10.0
                    
                    participant_preds.append(predicted_score)
                
                # Calculate average prediction for this participant
                avg_participant_pred = sum(participant_preds) / len(participant_preds)
                
                all_test_preds.append(avg_participant_pred)
                all_test_participant_ids.append(participant_id)
    
    # Convert test predictions to DataFrame
    test_pred_df = pd.DataFrame({
        'Participant_ID': all_test_participant_ids,
        'pred_score': all_test_preds
    })
    
    # Calculate average predicted score by participant ID (removing duplicates)
    test_avg_pred = test_pred_df.groupby('Participant_ID')['pred_score'].mean().reset_index()
    
    # Get true PHQ scores for test set
    test_true_phq = test_phq_df[['Participant_ID', 'PHQ8_Score']]
    
    # Merge predictions with true scores
    test_merged_df = pd.merge(test_avg_pred, test_true_phq, on='Participant_ID')
    
    # Calculate test evaluation metrics
    test_mse = mean_squared_error(test_merged_df['PHQ8_Score'], test_merged_df['pred_score'])
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_merged_df['PHQ8_Score'], test_merged_df['pred_score'])
    
    print(f"Test results - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Evaluated participants: {len(test_merged_df)}")
    
    # Save test predictions
    test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    print(f"Test predictions saved to {os.path.join(args.output_dir, 'test_predictions.csv')}")
    
    # Save final LoRA model
    final_lora_path = os.path.join(args.output_dir, "final_lora_model")
    model.save_pretrained(final_lora_path)
    print(f"Final LoRA model saved to {final_lora_path}")
    
    # Save final full model (merged)
    final_full_path = os.path.join(args.output_dir, "final_full_model")
    save_merged_model(model, tokenizer, final_full_path)
    print(f"Final full model saved to {final_full_path}")

def inference(model_path, text_input, use_full_model=False):
    """
    Use trained model to predict PHQ score for new text
    """
    # Load model and tokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_full_model:
        # Load the full merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
    else:
        # Load base model and apply LoRA adapters
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    
    # Prepare input
    prompt = f"Transcript: {text_input}\nPHQ Score:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate answer
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=1,
            do_sample=False,
            temperature=1.0
        )
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Extract predicted PHQ score
    try:
        phq_part = generated_text.split("PHQ Score:")[1].strip()
        predicted_score = float(''.join([c for c in phq_part if c.isdigit() or c == '.'][:5]))
    except:
        predicted_score = None
    
    return predicted_score, generated_text

if __name__ == "__main__":
    train()