import sys
import os
import importlib

# Print Python version and path
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Try to import torch and print information
print("\nAttempting to import torch...")
try:
    import torch
    
    # Get the path of the imported torch module
    print(f"Torch version: {torch.__version__}")
    print(f"Torch path: {torch.__file__}")
    
    # Check for basic torch attributes
    print("\nChecking basic torch attributes:")
    for attr in ['cuda', 'device', 'nn', 'optim']:
        has_attr = hasattr(torch, attr)
        print(f"  torch.{attr}: {'Available' if has_attr else 'NOT AVAILABLE'}")
    
    # Check if CUDA is available
    if hasattr(torch, 'cuda'):
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("\nCUDA module not available in torch")
        
except ImportError as e:
    print(f"Failed to import torch: {e}")
except Exception as e:
    print(f"Error while examining torch: {e}")

# Try to import torchtext and print information
print("\nAttempting to import torchtext...")
try:
    import torchtext
    
    # Get the path of the imported torchtext module
    print(f"Torchtext version: {torchtext.__version__}")
    print(f"Torchtext path: {torchtext.__file__}")
    
    # Check for legacy module
    has_legacy = hasattr(torchtext, 'legacy')
    print(f"  torchtext.legacy: {'Available' if has_legacy else 'NOT AVAILABLE'}")
    
except ImportError as e:
    print(f"Failed to import torchtext: {e}")
except Exception as e:
    print(f"Error while examining torchtext: {e}") 