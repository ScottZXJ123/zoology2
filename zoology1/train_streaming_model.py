import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
from torch.optim import AdamW

from zoology.config import DataConfig
from zoology.data.associative_recall import MQARConfig
from zoology.data.utils import prepare_data
from streaming_llm.enable_streaming_llm import enable_streaming_llm

def main():
    # 1. Configuration
    model_name = "EleutherAI/pythia-70m"
    start_size = 4
    recent_size = 256
    learning_rate = 1e-4
    num_epochs = 64
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Zoology MQAR Data
    print("Loading MQAR data...")
    data_config = DataConfig(
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=8192, input_seq_len=64, num_kv_pairs=8)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=8192, input_seq_len=64, num_kv_pairs=8)],
        batch_size=batch_size,
    )
    train_dataloader, test_dataloader = prepare_data(data_config)
    print("Data loaded.")

    # 3. Load and Modify Model with StreamingLLM
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply StreamingLLM modifications
    print("Applying StreamingLLM modifications...")
    enable_streaming_llm(model, start_size=4, recent_size=256)
    print("StreamingLLM enabled.")

    # 4. Training Loop
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    # 5. Evaluation
    print("Evaluating model...")
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy only on the query tokens (where target is not -100)
            mask = targets != -100
            total_correct += (predictions[mask] == targets[mask]).sum().item()
            total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 