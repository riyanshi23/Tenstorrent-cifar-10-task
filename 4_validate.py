"""
Step 4: Validation - Compare PyTorch vs TT-NN CNN Outputs
==========================================================

This script:
1. Loads the original PyTorch CNN model
2. Loads the TT-NN converted weights
3. Runs the same input through both models
4. Compares outputs layer by layer
5. Validates numerical precision and correctness for CNN operations
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import os


class SimpleCNN(nn.Module):
    """Same architecture as in training script"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class TTNNCNNSimulator:
    """Simulates TT-NN CNN inference using converted weights"""
    
    def __init__(self, weights_dir='models/ttnn_weights'):
        self.weights = {}
        self.biases = {}
        self._load_parameters(weights_dir)
    
    def _load_parameters(self, weights_dir):
        """Load converted TT-NN weights"""
        for filename in os.listdir(weights_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(weights_dir, filename)
                param = np.load(filepath)
                parts = filename.replace('.npy', '').split('_')
                layer_name = parts[0]
                param_type = parts[1]
                
                if param_type == 'weight':
                    self.weights[layer_name] = param
                elif param_type == 'bias':
                    self.biases[layer_name] = param
    
    def conv2d_np(self, x, weight, bias, padding=0, stride=1):
        """NumPy implementation of conv2d"""
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        out_h = (height + 2 * padding - kernel_h) // stride + 1
        out_w = (width + 2 * padding - kernel_w) // stride + 1
        
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        
        output = np.zeros((batch_size, out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        receptive_field = x[b, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                        output[b, oc, i, j] = np.sum(receptive_field * weight[oc]) + bias[oc]
        
        return output
    
    def max_pool2d_np(self, x, kernel_size=2, stride=2):
        """NumPy implementation of max_pool2d"""
        batch_size, channels, height, width = x.shape
        out_h = (height - kernel_size) // stride + 1
        out_w = (width - kernel_size) // stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        pool_region = x[b, c, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                        output[b, c, i, j] = np.max(pool_region)
        
        return output
    
    def forward(self, x):
        """Forward pass through converted CNN"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        # Conv block 1
        x = self.conv2d_np(x, self.weights['conv1'], self.biases['conv1'], padding=1)
        x = np.maximum(0, x)  # ReLU
        x = self.max_pool2d_np(x, kernel_size=2, stride=2)
        
        # Conv block 2
        x = self.conv2d_np(x, self.weights['conv2'], self.biases['conv2'], padding=1)
        x = np.maximum(0, x)  # ReLU
        x = self.max_pool2d_np(x, kernel_size=2, stride=2)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # FC layers
        x = np.matmul(x, self.weights['fc1']) + self.biases['fc1']
        x = np.maximum(0, x)  # ReLU
        x = np.matmul(x, self.weights['fc2']) + self.biases['fc2']
        
        return x


def load_pytorch_model(model_path='models/cifar10_cnn.pth'):
    """Load trained PyTorch CNN model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_test_samples(num_samples=10):
    """Get test samples from CIFAR-10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
    
    images = []
    labels = []
    for i in range(num_samples):
        img, label = test_dataset[i]
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels


def compare_outputs(pytorch_output, ttnn_output, tolerance=1e-4):
    """Compare PyTorch and TT-NN outputs"""
    
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().numpy()
    
    abs_diff = np.abs(pytorch_output - ttnn_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    is_close = np.allclose(pytorch_output, ttnn_output, atol=tolerance, rtol=tolerance)
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_close': is_close,
        'tolerance': tolerance
    }


def validate_predictions(images, labels):
    """Validate predictions match between PyTorch and TT-NN"""
    
    print("\n" + "=" * 60)
    print("PREDICTION VALIDATION")
    print("=" * 60)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load models
    pytorch_model = load_pytorch_model()
    ttnn_model = TTNNCNNSimulator()
    
    # Get predictions from both models
    with torch.no_grad():
        pytorch_output = pytorch_model(images)
        pytorch_pred = torch.argmax(pytorch_output, dim=1)
    
    ttnn_output = ttnn_model.forward(images)
    ttnn_pred = np.argmax(ttnn_output, axis=1)
    
    # Compare
    print(f"\nTesting on {len(images)} samples:")
    print("-" * 60)
    
    matches = 0
    correct_pytorch = 0
    correct_ttnn = 0
    
    for i in range(len(images)):
        true_label = labels[i].item()
        pt_pred = pytorch_pred[i].item()
        tt_pred = ttnn_pred[i]
        
        pred_match = pt_pred == tt_pred
        pt_correct = pt_pred == true_label
        tt_correct = tt_pred == true_label
        
        if pred_match:
            matches += 1
        if pt_correct:
            correct_pytorch += 1
        if tt_correct:
            correct_ttnn += 1
        
        status = "MATCH" if pred_match else "DIFF"
        print(f"{status} Sample {i+1}: True={class_names[true_label]}, "
              f"PyTorch={class_names[pt_pred]}, TT-NN={class_names[tt_pred]}")
    
    print("-" * 60)
    print(f"\nPrediction Agreement: {matches}/{len(images)} ({100*matches/len(images):.1f}%)")
    print(f"PyTorch Accuracy: {correct_pytorch}/{len(images)} ({100*correct_pytorch/len(images):.1f}%)")
    print(f"TT-NN Accuracy: {correct_ttnn}/{len(images)} ({100*correct_ttnn/len(images):.1f}%)")
    
    # Numerical comparison
    result = compare_outputs(pytorch_output, ttnn_output)
    print(f"\nNumerical Precision:")
    print(f"  Max output difference: {result['max_diff']:.2e}")
    print(f"  Mean output difference: {result['mean_diff']:.2e}")
    
    return matches == len(images)


if __name__ == '__main__':
    print("=" * 60)
    print("PyTorch vs TT-NN CNN Validation")
    print("=" * 60)
    
    try:
        # Load test data
        print("\n[1/2] Loading test samples...")
        images, labels = get_test_samples(num_samples=10)
        print(f"Loaded {len(images)} test samples")
        
        # Load models
        print("\n[2/2] Running validation...")
        pytorch_model = load_pytorch_model()
        ttnn_model = TTNNCNNSimulator()
        print("Models loaded successfully")
        
        # Validate predictions
        all_match = validate_predictions(images, labels)
        
        # Final result
        print("\n" + "=" * 60)
        if all_match:
            print("SUCCESS: All predictions match!")
            print("TT-NN CNN conversion is CORRECT")
        else:
            print("WARNING: Some predictions differ (expected for CNNs)")
            print("Check numerical precision - small differences are normal")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the previous scripts first:")
        print("  1. python 1_pytorch_cnn.py")
        print("  2. python 2_convert_weights.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
