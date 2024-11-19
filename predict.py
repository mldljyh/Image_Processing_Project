import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # Base CNN architecture for digit recognition
        # Input size: 28x28x1 (MNIST format)
        self.features = nn.Sequential(
            # Conv1: 28x28x1 -> 28x28x64 (with padding=1)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv2: 28x28x64 -> 26x26x128 (no padding)
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv3: 26x26x128 -> 26x26x256 (with padding=1)
            # followed by MaxPool: 26x26x256 -> 13x13x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.35),

            # Conv4: 13x13x256 -> 13x13x512 (with padding=1)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv5: 13x13x512 -> 11x11x1024 (no padding)
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Conv6: 11x11x1024 -> 11x11x2000 (with padding=1)
            # followed by MaxPool: 11x11x2000 -> 5x5x2000
            nn.Conv2d(1024, 2000, kernel_size=3, padding=1),
            nn.BatchNorm2d(2000),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.35),
        )

        # Final feature map size: 5x5x2000
        self.classifier = nn.Sequential(
            nn.Linear(2000 * 5 * 5, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 11)  # 11 classes (0-9 digits + no digit)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.features(x)  # Shape: (batch_size, 2000, 5, 5)
        features_flat = features.view(features.size(0), -1)  # Flatten: (batch_size, 2000*5*5)
        out = self.classifier(features_flat)
        return self.softmax(out), features

class FullyConnectedSubNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedSubNetwork, self).__init__()
        # Input size: 200x5x5 = 5000 (subset of features from BaseCNN)
        self.fc1 = nn.Sequential(
            nn.Linear(5000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 11)  # 11 classes (0-9 digits + no digit)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 200*5*5)
        x = self.fc1(x)
        x = nn.functional.dropout(self.fc2(x), p=0.5, training=self.training)
        x = self.fc3(x)
        return self.softmax(x)

class EnsNet(nn.Module):
    def __init__(self, num_subnets=10):
        super(EnsNet, self).__init__()
        # Ensemble network combining BaseCNN with multiple subnet classifiers
        self.base_cnn = BaseCNN()
        self.num_subnets = num_subnets
        self.subnets = nn.ModuleList([
            FullyConnectedSubNetwork() for _ in range(num_subnets)
        ])

    def forward(self, x):
        base_out, features = self.base_cnn(x)  # features: (batch_size, 2000, 5, 5)

        # Split 2000 channels into num_subnets parts (200 channels each)
        split_features = torch.chunk(features, self.num_subnets, dim=1)

        # Get predictions from each subnet
        subnet_outputs = []
        for i, subnet in enumerate(self.subnets):
            subnet_outputs.append(subnet(split_features[i]))

        # Stack all outputs including base_out
        # Shape: (batch_size, num_subnets+1, num_classes)
        all_outputs = torch.stack([base_out] + subnet_outputs, dim=1)

        # Ensemble prediction (average)
        ensemble_out = torch.mean(all_outputs, dim=1)

        return ensemble_out, all_outputs

def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def preprocess_image(image_path):
    image = np.array(Image.open(image_path))
    image_array = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    binary_image = 255 - ((image_array > 100).astype(np.uint8) * 255)
    image = Image.fromarray(binary_image)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        ensemble_out, _ = model(image_tensor)
        probabilities = ensemble_out[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        if predicted_class == 10:
            return "no digit", confidence
        return str(predicted_class), confidence

def main():
    checkpoint_path = 'checkpoints/best_model.pth'
    model, device = load_model(checkpoint_path)
    
    cell_dir = './cell'
    supported_formats = ('.png', '.jpg', '.jpeg')
    
    def get_cell_number(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except:
            return float('inf')

    image_files = [f for f in os.listdir(cell_dir) 
                  if f.lower().endswith(supported_formats)]
    image_files.sort(key=get_cell_number)
    
    grid = [['' for _ in range(7)] for _ in range(7)]
    
    for image_path in image_files:
        cell_num = int(image_path.split('_')[1].split('.')[0])
        row = cell_num // 7
        col = cell_num % 7
        
        image_tensor = preprocess_image(os.path.join(cell_dir, image_path))
        prediction, confidence = predict(model, image_tensor, device)
        grid[row][col] = prediction if prediction != "no digit" else "X"

    print("\n7x7 Grid:")
    print("-" * 29)
    for i, row in enumerate(grid):
        print("|", end=" ")
        for j, val in enumerate(row):
            print(f"{val:^2}", end=" ")
            if (j + 1) % 7 == 0:
                print("|")
        if (i + 1) % 7 == 0:
            print("-" * 29)

if __name__ == "__main__":
    main()