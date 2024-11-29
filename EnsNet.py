import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
from tqdm import tqdm
import mkl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Enable multi-core processing for NumPy
mkl.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # MNIST input: 28x28x1
        self.features = nn.Sequential(
            # 28x28x1 -> 28x28x64 (padding=1)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.35),

            # 28x28x64 -> 26x26x128 (no padding)
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            # 26x26x128 -> 26x26x256 (padding=1)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 26x26x256 -> 13x13x256 (maxpool)
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.35),

            # 13x13x256 -> 13x13x512 (padding=1)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            # 13x13x512 -> 11x11x1024 (no padding)
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            # 11x11x1024 -> 11x11x2000 (padding=1)
            nn.Conv2d(1024, 2000, kernel_size=3, padding=1),
            nn.BatchNorm2d(2000),
            nn.ReLU(),
            # 11x11x2000 -> 5x5x2000 (maxpool)
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.35),
        )

        # Final feature map size: 5x5x2000
        self.classifier = nn.Sequential(
            nn.Linear(2000 * 5 * 5, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 11)  # Changed from 10 to 11 classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.features(x)  # (batch_size, 2000, 5, 5)
        features_flat = features.view(features.size(0), -1)  # (batch_size, 2000*5*5)
        out = self.classifier(features_flat)
        return self.softmax(out), features

class FullyConnectedSubNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedSubNetwork, self).__init__()
        # Input: 200x5x5 = 5000
        self.fc1 = nn.Sequential(
            nn.Linear(5000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11)  # Changed from 10 to 11 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, 200*5*5)
        x = self.fc1(x)
        x = nn.functional.dropout(self.fc2(x), p=0.5, training=self.training)
        x = self.fc3(x)
        return self.softmax(x)

class EnsNet(nn.Module):
    def __init__(self, num_subnets=10):
        super(EnsNet, self).__init__()
        self.base_cnn = BaseCNN()
        self.num_subnets = num_subnets
        self.subnets = nn.ModuleList([
            FullyConnectedSubNetwork() for _ in range(num_subnets)
        ])

    def forward(self, x):
        base_out, features = self.base_cnn(x)  # features: (batch_size, 2000, 5, 5)

        # Split 2000 channels into 10 subnets (200 channels each)
        split_features = torch.chunk(features, self.num_subnets, dim=1)

        # Predictions from each subnet
        subnet_outputs = []
        for i, subnet in enumerate(self.subnets):
            subnet_outputs.append(subnet(split_features[i]))

        # Stack all outputs (including base_out)
        # (batch_size, num_subnets+1, num_classes)
        all_outputs = torch.stack([base_out] + subnet_outputs, dim=1)

        # Ensemble prediction (average)
        ensemble_out = torch.mean(all_outputs, dim=1)

        return ensemble_out, all_outputs

# Add this new class before the data loading section
class ExtendedMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, num_no_digit=6000):
        self.mnist_dataset = mnist_dataset
        self.num_no_digit = num_no_digit
        
        # Create completely black images
        self.no_digit_images = torch.zeros(num_no_digit, 1, 28, 28)
        
        # Add extremely subtle random noise (values between 0 and 0.02)
        noise_mask = torch.rand(num_no_digit, 1, 28, 28) < 0.1  # Only 10% of pixels get noise
        self.no_digit_images[noise_mask] += torch.rand(noise_mask.sum()) * 0.02
        
        # Store labels as a tensor
        self.no_digit_labels = torch.full((num_no_digit,), 10, dtype=torch.long)
        
    def __len__(self):
        return len(self.mnist_dataset) + self.num_no_digit
    
    def __getitem__(self, idx):
        if idx < len(self.mnist_dataset):
            image, label = self.mnist_dataset[idx]
            return image, torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor
        else:
            noise_idx = idx - len(self.mnist_dataset)
            img = self.no_digit_images[noise_idx].clone()
            
            # Occasionally add minimal random variation
            if torch.rand(1) > 0.7:  # 30% chance
                noise_mask = torch.rand_like(img) < 0.1
                img[noise_mask] += torch.rand(noise_mask.sum()) * 0.02
            
            return img, self.no_digit_labels[noise_idx].clone()  # Return cloned tensor

# Data Augmentaiton
transform = transforms.Compose([
    # First pad the image to 32x32 with black padding (0)
    transforms.Pad(padding=2, fill=0),  # 28x28 -> 32x32
    transforms.RandomAffine(
        degrees=30,  # Slightly more rotation
        scale=(0.5, 1.1),  # Allow smaller scales but don't make numbers larger
        translate=(0.25, 0.25),  # Increase translation range
        shear=0.2,  # Reduce shear to prevent excessive distortion
        fill=0,  # Fill empty space with black
    ),
    # Center crop back to 28x28 - this ensures no number gets cut off
    transforms.CenterCrop(28),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
])

# For test set, just use padding to keep original scale reference
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Data Loader
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=test_transform)

extended_train = ExtendedMNIST(mnist_train, num_no_digit=12000)  # Changed from 6000 to 12000 (20% of MNIST training set)
extended_test = ExtendedMNIST(mnist_test, num_no_digit=2000)    # Changed from 1000 to 2000 (20% of MNIST test set)

train_loader = DataLoader(extended_train, batch_size=2048, shuffle=True, num_workers=10, 
                         pin_memory=True, persistent_workers=True)
test_loader = DataLoader(extended_test, batch_size=2048, shuffle=False, num_workers=10,
                        pin_memory=True, persistent_workers=True)

# Add new Scheduler class
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, warmup_start_lr=1e-8):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnsNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=1300, base_lr=0.0002)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(model, optimizer, epoch, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy = checkpoint['accuracy']
        print(f"Checkpoint loaded: {filename}")
        print(f"Resumed from epoch {epoch} with accuracy {accuracy:.2f}%")
        return epoch, accuracy
    return 0, 0


# Initialize wandb
wandb.init(
    project="mnist-ensemble",
    config={
        "architecture": "EnsNet",
        "dataset": "Extended-MNIST",
        "epochs": 1300,
        "batch_size": 512,
        "base_lr": 0.0002,
        "warmup_epochs": 5,
        "weight_decay": 0.01,
        "num_subnets": 10,
        "optimizer": "AdamW",
        "num_classes": 11
    }
)



def log_confusion_matrix(y_true, y_pred, epoch):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    plt.figure(figsize=(11, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Log to wandb
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

def calculate_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc

def train(epochs=1300, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_accuracy = 0
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, checkpoint_path)

    # Create epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training")

    for epoch in epoch_pbar:
        # Get and log current learning rate
        current_lr = scheduler.step()
        epoch_metrics = {
            "learning_rate": current_lr,
            "epoch": epoch
        }
        
        model.train()
        running_loss = 0.0
        batch_losses = []

        # Create batch progress bar
        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}")

        for batch_idx, (data, target) in enumerate(batch_pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            ensemble_out, all_outputs = model(data)

            loss = criterion(ensemble_out, target)
            for i in range(all_outputs.size(1)):
                subnet_out = all_outputs[:, i, :]
                loss += criterion(subnet_out, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())

            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

        # Calculate average epoch loss
        epoch_loss = running_loss / len(train_loader)
        epoch_metrics.update({
            "train_loss": epoch_loss,
            "batch_losses_mean": np.mean(batch_losses),
            "batch_losses_std": np.std(batch_losses)
        })

        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    ensemble_out, all_outputs = model(data)
                    _, predicted = torch.max(ensemble_out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    test_loss += criterion(ensemble_out, target).item()
                    
                    # Collect predictions and targets
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

                    # Calculate subnet losses
                    subnet_losses = [criterion(all_outputs[:, i, :], target).item() 
                                   for i in range(all_outputs.size(1))]

                current_accuracy = 100 * correct / total
                avg_test_loss = test_loss / len(test_loader)

                # Calculate and log per-class accuracy
                per_class_acc = calculate_per_class_accuracy(all_targets, all_preds)
                per_class_metrics = {f"class_{i}_accuracy": acc * 100 
                                   for i, acc in enumerate(per_class_acc)}
                
                # Log confusion matrix
                log_confusion_matrix(all_targets, all_preds, epoch)
                
                # Enhanced wandb logging with epoch-based metrics
                epoch_metrics.update({
                    "test_loss": avg_test_loss,
                    "test_accuracy": current_accuracy,
                    **per_class_metrics,
                    "base_subnet_loss": subnet_losses[0],
                    "avg_subnet_loss": sum(subnet_losses[1:]) / len(subnet_losses[1:])
                })

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    save_checkpoint(model, optimizer, epoch, current_accuracy, best_model_path)
                    epoch_metrics["best_accuracy"] = best_accuracy

        # Log all metrics for the epoch at once
        wandb.log(epoch_metrics)
        model.train()

def evaluate(model_path):
    load_checkpoint(model, optimizer, model_path)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    # Add progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc="Evaluating")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in eval_pbar:
            data, target = data.to(device), target.to(device)
            ensemble_out, _ = model(data)
            _, predicted = torch.max(ensemble_out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss += criterion(ensemble_out, target).item()

            # Update progress bar
            current_accuracy = 100 * correct / total
            eval_pbar.set_postfix({
                'accuracy': f'{current_accuracy:.2f}%'
            })
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    final_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    # Calculate and log final metrics
    per_class_acc = calculate_per_class_accuracy(all_targets, all_preds)
    per_class_metrics = {f"final_class_{i}_accuracy": acc * 100 
                        for i, acc in enumerate(per_class_acc)}
    
    # Log final confusion matrix
    log_confusion_matrix(all_targets, all_preds, "Final")

    # Enhanced final wandb logging
    wandb.log({
        "final_test_accuracy": final_accuracy,
        "final_test_loss": avg_test_loss,
        **per_class_metrics,
    })

    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    for i, acc in enumerate(per_class_acc):
        print(f'Class {i} Accuracy: {acc*100:.2f}%')
    
    return final_accuracy

if __name__ == "__main__":
    train()
    best_model_path = os.path.join('checkpoints', 'best_model.pth')
    final_accuracy = evaluate(best_model_path)

    # Close wandb run
    wandb.finish()