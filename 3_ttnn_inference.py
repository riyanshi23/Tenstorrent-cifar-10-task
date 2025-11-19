"""
Step 3: TT-NN Inference Implementation (Simulated with NumPy)

This script demonstrates how to perform inference using the converted TT-NN weights.
Since we don't have actual Tenstorrent hardware, we simulate the operations using NumPy.

In a real TT-NN implementation, we would replace these NumPy operations with TT-NN API calls.
"""

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def conv2d(input_data, weights, bias, stride=1, padding=1):
    """
    Simplified 2D convolution (NumPy simulation)
    
    Args:
        input_data: [batch, in_channels, height, width]
        weights: [out_channels, in_channels, kernel_h, kernel_w]
        bias: [out_channels]
        stride: stride value
        padding: padding value
    
    Returns:
        output: [batch, out_channels, out_height, out_width]
    """
    batch_size, in_channels, in_h, in_w = input_data.shape
    out_channels, _, k_h, k_w = weights.shape
    
    # Add padding
    if padding > 0:
        input_data = np.pad(input_data, 
                           ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                           mode='constant', constant_values=0)
    
    # Calculate output dimensions
    out_h = (in_h + 2*padding - k_h) // stride + 1
    out_w = (in_w + 2*padding - k_w) // stride + 1
    
    output = np.zeros((batch_size, out_channels, out_h, out_w))
    
    # Perform convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + k_h
                    w_end = w_start + k_w
                    
                    receptive_field = input_data[b, :, h_start:h_end, w_start:w_end]
                    output[b, oc, i, j] = np.sum(receptive_field * weights[oc]) + bias[oc]
    
    return output

def maxpool2d(input_data, kernel_size=2, stride=2):
    """
    Max pooling operation
    
    Args:
        input_data: [batch, channels, height, width]
        kernel_size: size of pooling window
        stride: stride value
    
    Returns:
        output: [batch, channels, out_height, out_width]
    """
    batch_size, channels, in_h, in_w = input_data.shape
    
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1
    
    output = np.zeros((batch_size, channels, out_h, out_w))
    
    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    
                    pool_region = input_data[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, i, j] = np.max(pool_region)
    
    return output

def linear(x, weight, bias):
    """
    Linear layer: output = x @ weight + bias
    
    Args:
        x: [batch, in_features]
        weight: [in_features, out_features] (TT-NN format)
        bias: [out_features]
    
    Returns:
        output: [batch, out_features]
    """
    return np.dot(x, weight) + bias

def ttnn_cnn_inference(input_image):
    """
    Perform CNN inference using TT-NN weights (simulated with NumPy)
    
    Args:
        input_image: [batch, 3, 32, 32] normalized input
    
    Returns:
        logits: [batch, 10] class scores
    """
    # Load TT-NN weights
    conv1_weight = np.load('models/ttnn_weights/conv1_weight.npy')
    conv1_bias = np.load('models/ttnn_weights/conv1_bias.npy')
    conv2_weight = np.load('models/ttnn_weights/conv2_weight.npy')
    conv2_bias = np.load('models/ttnn_weights/conv2_bias.npy')
    fc1_weight = np.load('models/ttnn_weights/fc1_weight.npy')
    fc1_bias = np.load('models/ttnn_weights/fc1_bias.npy')
    fc2_weight = np.load('models/ttnn_weights/fc2_weight.npy')
    fc2_bias = np.load('models/ttnn_weights/fc2_bias.npy')
    
    # Forward pass
    # Conv block 1: conv -> relu -> pool
    x = conv2d(input_image, conv1_weight, conv1_bias, stride=1, padding=1)
    x = relu(x)
    x = maxpool2d(x, kernel_size=2, stride=2)  # 32x32 -> 16x16
    
    # Conv block 2: conv -> relu -> pool
    x = conv2d(x, conv2_weight, conv2_bias, stride=1, padding=1)
    x = relu(x)
    x = maxpool2d(x, kernel_size=2, stride=2)  # 16x16 -> 8x8
    
    # Flatten
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1)  # [batch, 64*8*8]
    
    # Fully connected layers
    x = linear(x, fc1_weight, fc1_bias)
    x = relu(x)
    x = linear(x, fc2_weight, fc2_bias)
    
    return x

def test_ttnn_inference():
    """Test TT-NN inference on CIFAR-10 test set"""
    
    print("Loading CIFAR-10 test dataset...")
    
    # Data transformation (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Test on first 100 samples
    print("\nRunning TT-NN inference on 100 test samples...")
    correct = 0
    total = 0
    
    for i in range(100):
        image, label = test_dataset[i]
        
        # Prepare input: add batch dimension
        input_image = image.numpy()[np.newaxis, ...]  # [1, 3, 32, 32]
        
        # Run TT-NN inference
        logits = ttnn_cnn_inference(input_image)
        
        # Get prediction
        prediction = np.argmax(logits[0])
        
        if prediction == label:
            correct += 1
        total += 1
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/100 images...")
    
    accuracy = 100 * correct / total
    print(f"\nTT-NN Inference Accuracy (100 samples): {accuracy:.2f}%")
    
    # Visualize some predictions
    visualize_ttnn_predictions(test_dataset)

def visualize_ttnn_predictions(test_dataset):
    """Visualize TT-NN predictions on sample images"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(10):
        image, label = test_dataset[idx]
        
        # Run TT-NN inference
        input_image = image.numpy()[np.newaxis, ...]
        logits = ttnn_cnn_inference(input_image)
        prediction = np.argmax(logits[0])
        
        # Denormalize for visualization
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
        img = image.numpy() * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'True: {classes[label]}\nPred: {classes[prediction]}',
                           color='green' if prediction == label else 'red')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/ttnn_predictions.png')
    print("TT-NN predictions visualization saved to 'models/ttnn_predictions.png'")

if __name__ == '__main__':
    test_ttnn_inference()
    print("\nStep 3 complete! TT-NN inference tested.")
    print("\n" + "="*70)
    print("NOTE: This is a NumPy simulation of TT-NN operations.")
    print("On actual Tenstorrent hardware, you would use the TT-NN API:")
    print("  - ttnn.conv2d() for convolution")
    print("  - ttnn.max_pool2d() for pooling")
    print("  - ttnn.matmul() for linear layers")
    print("  - ttnn.relu() for activation")
    print("="*70)
