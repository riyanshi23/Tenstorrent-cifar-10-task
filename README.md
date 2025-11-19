A PyTorch CNN trained on CIFAR-10 (10-class image classification) and converted to TensorTorrent Neural Network (TT-NN) format. 
The project includes 4 scripts: 
(1) training a simple CNN with 2 conv layers and 2 FC layers, 
(2) extracting and converting PyTorch weights to `.npy` format, 
(3) implementing TT-NN inference operations in NumPy (simulated), and 
(4) validating that both models produce identical predictions. 

Run `python run_all.py` to execute the complete pipeline, or run individual scripts (`1_pytorch_cnn.py`, `2_convert_weights.py`, `3_ttnn_inference.py`, `4_validate.py`) separately. 
Achieves ~70% accuracy on CIFAR-10 test set and demonstrates end-to-end model conversion for specialized AI hardware.

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
