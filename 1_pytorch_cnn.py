"""
Step 1: Build and Train PyTorch CNN for CIFAR-10
=================================================

This script:
1. Loads the CIFAR-10 dataset (32x32 RGB images)
2. Defines a Convolutional Neural Network architecture
3. Trains the CNN model
4. Saves the trained weights for later conversion to TT-NN

Architecture:
- Conv2d(3, 32, 3x3) + ReLU + MaxPool(2x2)
- Conv2d(32, 64, 3x3) + ReLU + MaxPool(2x2)
- Flatten
- Linear(64*7*7, 512) + ReLU
- Linear(512, 10)

CIFAR-10 Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class SimpleCNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10 classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 pooling layers: 32x32 → 16x16 → 8x8
        # Feature map size: 64 * 8 * 8 = 4096
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
        self.layer_info = {
            'conv1': {'in': 3, 'out': 32, 'kernel': 3, 'padding': 1},
            'conv2': {'in': 32, 'out': 64, 'kernel': 3, 'padding': 1},
            'fc1': {'in': 4096, 'out': 512},
            'fc2': {'in': 512, 'out': 10}
        }
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)      # [batch, 32, 32, 32]
        x = self.relu1(x)
        x = self.pool1(x)      # [batch, 32, 16, 16]
        
        # Conv block 2
        x = self.conv2(x)      # [batch, 64, 16, 16]
        x = self.relu2(x)
        x = self.pool2(x)      # [batch, 64, 8, 8]
        
        # Fully connected layers
        x = self.flatten(x)    # [batch, 4096]
        x = self.fc1(x)        # [batch, 512]
        x = self.relu3(x)
        x = self.fc2(x)        # [batch, 10]
        
        return x


def load_cifar10_data(batch_size=64):
    """Load and preprocess CIFAR-10 dataset"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Download and load training data
    train_dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Download and load test data
    test_dataset = datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    """Train the CNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    print(f"Training on {device}")
    print(f"Model architecture: {model.layer_info}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                            'acc': f'{100 * correct / total:.2f}%'})
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def save_model(model, filepath='models/cifar10_cnn.pth'):
    """Save trained model weights"""
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer_info': model.layer_info
    }, filepath)
    print(f"\nModel saved to {filepath}")


def visualize_predictions(model, test_loader, class_names, num_samples=10):
    """Visualize some predictions"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predictions[i].item()]
            color = 'green' if labels[i] == predictions[i] else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/predictions.png')
    print("Sample predictions saved to models/predictions.png")


if __name__ == '__main__':
    print("=" * 60)
    print("CIFAR-10 CNN Training with PyTorch")
    print("=" * 60)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load data
    print("\n[1/4] Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n[2/4] Creating CNN model...")
    model = SimpleCNN(num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n[3/4] Training model...")
    history = train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001)
    
    # Save model
    print("\n[4/4] Saving model...")
    save_model(model)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, class_names)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("=" * 60)
