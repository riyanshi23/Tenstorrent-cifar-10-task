"""
Step 2: Convert PyTorch CNN Weights to TT-NN Format
====================================================

This script:
1. Loads the trained PyTorch CNN model
2. Extracts weights from convolutional and linear layers
3. Converts them to the format required by TT-NN
4. Handles layout conversions (NCHW → NHWC for convolutions)
5. Saves converted weights for TT-NN inference

Key conversions:
- Conv2d weights: [out_channels, in_channels, H, W] → may need layout adjustments
- Linear weights: [out, in] → [in, out] (transpose)
- Bias vectors: Convert to appropriate format
"""

import torch
import numpy as np
import os
import json


def load_pytorch_model(model_path='models/cifar10_cnn.pth'):
    """Load the trained PyTorch CNN model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    layer_info = checkpoint['layer_info']
    
    print(f"Loaded model from {model_path}")
    print(f"Architecture: {layer_info}")
    
    return state_dict, layer_info


def extract_layer_parameters(state_dict):
    """Extract weights and biases from PyTorch state dict"""
    
    layers = {}
    
    print("\nExtracting parameters:")
    for key, value in state_dict.items():
        # Parse layer name and parameter type
        parts = key.split('.')
        layer_name = parts[0]  # e.g., 'conv1', 'fc1'
        param_type = parts[1]   # 'weight' or 'bias'
        
        if layer_name not in layers:
            layers[layer_name] = {}
        
        # Convert to numpy
        layers[layer_name][param_type] = value.numpy()
        
        print(f"  {key}: shape {value.shape}")
    
    return layers


def convert_conv_weights(layers):
    """
    Convert Conv2d weights for TT-NN
    
    PyTorch Conv2d weight format: [out_channels, in_channels, kernel_H, kernel_W]
    - Uses NCHW (batch, channels, height, width) layout
    
    TT-NN may expect different layouts depending on the operation:
    - Some operations use NHWC (batch, height, width, channels)
    - Weight format might need adjustment
    
    For this conversion, we'll keep weights in PyTorch format but document
    the layout considerations for actual TT-NN deployment.
    """
    
    converted_layers = {}
    
    for layer_name, params in layers.items():
        if 'conv' in layer_name:
            converted_layers[layer_name] = {}
            
            if 'weight' in params:
                weight = params['weight']  # [out, in, H, W]
                
                # For TT-NN, depending on the API, you might need to:
                # 1. Keep as-is if TT-NN accepts NCHW format
                # 2. Permute to NHWC format: weight.transpose(0, 2, 3, 1)
                # 3. Or other layout transformations
                
                # For now, we'll keep the original format and add metadata
                converted_layers[layer_name]['weight'] = weight
                converted_layers[layer_name]['weight_format'] = 'OIHW'  # [out, in, H, W]
                
                print(f"\n{layer_name} conv weight:")
                print(f"  Shape: {weight.shape} [out_ch, in_ch, H, W]")
                print(f"  Format: OIHW (PyTorch standard)")
            
            if 'bias' in params:
                converted_layers[layer_name]['bias'] = params['bias']
                print(f"  Bias: {params['bias'].shape}")
    
    return converted_layers


def convert_linear_weights(layers):
    """
    Convert Linear layer weights for TT-NN
    
    Same as MNIST task:
    - PyTorch: weight shape [out, in], uses output = input @ weight.T + bias
    - TT-NN: weight shape [in, out], uses output = input @ weight + bias
    
    Solution: Transpose the weights
    """
    
    converted_layers = {}
    
    for layer_name, params in layers.items():
        if 'fc' in layer_name:
            converted_layers[layer_name] = {}
            
            if 'weight' in params:
                pytorch_weight = params['weight']  # [out, in]
                ttnn_weight = pytorch_weight.T      # [in, out]
                converted_layers[layer_name]['weight'] = ttnn_weight
                
                print(f"\n{layer_name} linear weight conversion:")
                print(f"  PyTorch shape: {pytorch_weight.shape} [out, in]")
                print(f"  TT-NN shape: {ttnn_weight.shape} [in, out]")
            
            if 'bias' in params:
                converted_layers[layer_name]['bias'] = params['bias']
                print(f"  Bias: {params['bias'].shape}")
    
    return converted_layers


def save_converted_weights(conv_layers, linear_layers, output_dir='models/ttnn_weights'):
    """Save converted weights in numpy format"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_layers = {**conv_layers, **linear_layers}
    
    # Save each layer's parameters
    for layer_name, params in all_layers.items():
        for param_name, param_value in params.items():
            if param_name in ['weight', 'bias']:  # Skip metadata fields
                filename = f"{layer_name}_{param_name}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, param_value)
                print(f"Saved: {filepath}")
    
    # Save metadata
    metadata = {
        'conv_layers': list(conv_layers.keys()),
        'linear_layers': list(linear_layers.keys()),
        'conversion_notes': {
            'conv_weights': 'OIHW format [out_channels, in_channels, H, W]',
            'linear_weights': 'Transposed for TT-NN [in_features, out_features]',
            'layout_consideration': 'TT-NN may require NHWC layout for convolutions',
            'bias_format': 'Same as PyTorch',
            'data_type': 'float32'
        },
        'layer_details': {
            'conv1': {'in_channels': 3, 'out_channels': 32, 'kernel': 3, 'padding': 1},
            'conv2': {'in_channels': 32, 'out_channels': 64, 'kernel': 3, 'padding': 1},
            'fc1': {'in_features': 4096, 'out_features': 512},
            'fc2': {'in_features': 512, 'out_features': 10}
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved: {metadata_path}")


def verify_conversion(original_layers, conv_layers, linear_layers):
    """Verify that conversion was done correctly"""
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Verify conv layers (weights kept as-is, just extracted)
    for layer_name in conv_layers.keys():
        if 'weight' in original_layers[layer_name]:
            orig_weight = original_layers[layer_name]['weight']
            conv_weight = conv_layers[layer_name]['weight']
            
            assert np.array_equal(orig_weight, conv_weight), \
                f"{layer_name}: Conv weight changed unexpectedly"
            
            print(f"✓ {layer_name}: Conv weights correctly preserved")
        
        if 'bias' in original_layers[layer_name]:
            orig_bias = original_layers[layer_name]['bias']
            conv_bias = conv_layers[layer_name]['bias']
            
            assert np.array_equal(orig_bias, conv_bias), \
                f"{layer_name}: Bias changed unexpectedly"
            
            print(f"✓ {layer_name}: Bias correctly preserved")
    
    # Verify linear layers (weights transposed)
    for layer_name in linear_layers.keys():
        if 'weight' in original_layers[layer_name]:
            orig_weight = original_layers[layer_name]['weight']
            lin_weight = linear_layers[layer_name]['weight']
            
            assert orig_weight.shape[0] == lin_weight.shape[1], \
                f"{layer_name}: Dimension mismatch after transpose"
            assert orig_weight.shape[1] == lin_weight.shape[0], \
                f"{layer_name}: Dimension mismatch after transpose"
            
            assert np.allclose(orig_weight.T, lin_weight), \
                f"{layer_name}: Transpose not done correctly"
            
            print(f"✓ {layer_name}: Linear weights correctly transposed")
        
        if 'bias' in original_layers[layer_name]:
            orig_bias = original_layers[layer_name]['bias']
            lin_bias = linear_layers[layer_name]['bias']
            
            assert np.array_equal(orig_bias, lin_bias), \
                f"{layer_name}: Bias changed unexpectedly"
            
            print(f"✓ {layer_name}: Bias correctly preserved")
    
    print("\n✅ All conversions verified successfully!")


if __name__ == '__main__':
    print("=" * 60)
    print("Converting PyTorch CNN Weights to TT-NN Format")
    print("=" * 60)
    
    # Load PyTorch model
    print("\n[1/4] Loading PyTorch CNN model...")
    state_dict, layer_info = load_pytorch_model()
    
    # Extract parameters
    print("\n[2/4] Extracting layer parameters...")
    layers = extract_layer_parameters(state_dict)
    
    # Convert convolutional layers
    print("\n[3/4] Converting weights...")
    print("\n--- Convolutional Layers ---")
    conv_layers = convert_conv_weights(layers)
    
    print("\n--- Linear Layers ---")
    linear_layers = convert_linear_weights(layers)
    
    # Save converted weights
    print("\n[4/4] Saving converted weights...")
    save_converted_weights(conv_layers, linear_layers)
    
    # Verify conversion
    verify_conversion(layers, conv_layers, linear_layers)
    
    print("\n" + "=" * 60)
    print("Weight Conversion Complete!")
    print("=" * 60)
    print("\nConverted weights saved in: models/ttnn_weights/")
    print("Ready for TT-NN inference!")
