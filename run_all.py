"""
Run all steps in sequence:
1. Train PyTorch CNN (if not already trained)
2. Convert weights to TT-NN format
3. Run TT-NN inference
4. Validate results
"""

import os
import subprocess
import sys


def run_script(script_name, description):
    """Run a Python script"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n[SUCCESS] {description} - COMPLETED")
        return True
    except Exception as e:
        print(f"\n[FAILED] {description} - FAILED: {e}")
        return False


def check_model_exists():
    """Check if model is already trained"""
    model_path = os.path.join('models', 'cifar10_cnn.pth')
    return os.path.exists(model_path)


def main():
    print("=" * 70)
    print("CIFAR-10 CNN → TT-NN COMPLETE PIPELINE")
    print("=" * 70)
    
    # Check if we need to train
    if check_model_exists():
        print("Found existing trained model")
        response = input("Retrain? (y/n): ").lower().strip()
        skip_training = response != 'y'
    else:
        print("No trained model found - will train from scratch")
        skip_training = False
    
    input("\nPress Enter to start...")
    
    results = []
    
    # Step 1: Train (optional)
    if not skip_training:
        success = run_script('1_pytorch_cnn.py', 'Step 1: Train PyTorch CNN')
        results.append(('Train CNN', success))
    else:
        print("\n" + "=" * 70)
        print("SKIPPING: Step 1 - Using existing model")
        print("=" * 70)
        results.append(('Train CNN', True))
    
    # Step 2: Convert
    success = run_script('2_convert_weights.py', 'Step 2: Convert Weights')
    results.append(('Convert Weights', success))
    if not success:
        return
    
    # Step 3: Inference
    success = run_script('3_ttnn_inference.py', 'Step 3: TT-NN Inference')
    results.append(('TT-NN Inference', success))
    
    # Step 4: Validate
    success = run_script('4_validate.py', 'Step 4: Validate')
    results.append(('Validation', success))
    
    # Summary
    print("\n\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    for step_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10} - {step_name}")
    
    if all(success for _, success in results):
        print("\nSUCCESS! All steps completed!")
    else:
        print("\nSome steps failed.")


if __name__ == '__main__':
    main()
