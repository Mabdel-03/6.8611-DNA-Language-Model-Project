import os
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from datasets import load_dataset

# Constants
SEQUENCE_LENGTH = 114688  # 2**17
TASK_NAME = "cage_prediction"
REFERENCE_GENOME_PATH = "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Downloads/cache/downloads/hg38.fa.gz"
CACHE_DIR = "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Downloads/cache"

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=SEQUENCE_LENGTH,
    data_files={"reference_genome": REFERENCE_GENOME_PATH},
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

class DNADataset(Dataset):
    def __init__(self, dataset, sequence_length, mapping):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_sequence = self.dataset[idx]['sequence']
        sequence = torch.tensor([self.mapping[base] for base in raw_sequence], dtype=torch.long)
        sequence = sequence[:self.sequence_length]
        one_hot_sequence = seq_indices_to_one_hot(sequence.unsqueeze(0)).squeeze(0)
        target = torch.tensor(self.dataset[idx]['labels'][0]).float()
        return one_hot_sequence, target.unsqueeze(0)

# Initialize Enformer model
model = Enformer.from_hparams(
    dim=1536,
    depth=11,
    heads=8,
    output_heads=dict(human=50),  # Full set of human targets
    target_length=896,
)

# Wrap model with DataParallel
model = DataParallel(model)
model = model.cuda()

mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4}

# Hyperparameters
batch_size = 64 * torch.cuda.device_count()  # Increase batch size for multiple GPUs
num_epochs = 50
learning_rate = 1e-3

# DataLoaders
train_loader = DataLoader(
    DNADataset(train_dataset, SEQUENCE_LENGTH, mapping),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 * torch.cuda.device_count()
)

val_loader = DataLoader(
    DNADataset(val_dataset, SEQUENCE_LENGTH, mapping),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4 * torch.cuda.device_count()
)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

# Training and Validation Loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_idx, (one_hot_sequence, target) in enumerate(train_loader):
        one_hot_sequence = one_hot_sequence.cuda()
        target = target.cuda()

        # Forward pass
        output = model(one_hot_sequence, head='human')
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for one_hot_sequence, target in val_loader:
            one_hot_sequence = one_hot_sequence.cuda()
            target = target.cuda()
            output = model(one_hot_sequence, head='human')
            val_loss = criterion(output, target)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f"/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Model/checkpoint_epoch_{epoch+1}.pth")

print("Training Complete")

# Save final model
torch.save(model.module.state_dict(), "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Model/Trained_Model_OG_Params.pth")
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
}, "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Model/Trained_Model_OG_State.pth")
