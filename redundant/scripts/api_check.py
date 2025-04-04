#!/usr/bin/env python
"""
Check which torchtext API is being used and display information about the environment.

This script helps users verify whether they are using the legacy or modern API
and confirms that the necessary components are available.
"""
import os
import sys
import importlib
import warnings

print("===== Torchtext API Check =====\n")

# Check for torchtext
print("Checking torchtext installation...")
try:
    import torchtext
    print(f"✓ Torchtext version: {torchtext.__version__}")
    print(f"  Path: {torchtext.__file__}")
except ImportError:
    print("✗ Torchtext is not installed")
    print("  Run: pip install torchtext==0.15.0")
    sys.exit(1)

# Check for torchtext.legacy
print("\nChecking for torchtext.legacy...")
try:
    from torchtext.legacy.data import Field
    print("✓ torchtext.legacy.data.Field is available")
    print("  You are using the LEGACY API")
except ImportError:
    print("✗ torchtext.legacy.data.Field is NOT available")
    print("  You are using the MODERN API")

# Check for PyTorch
print("\nChecking PyTorch installation...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  Path: {torch.__file__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA is not available - using CPU only")
except ImportError:
    print("✗ PyTorch is not installed")
    print("  Run: pip install torch==2.0.0")
    sys.exit(1)

# Check for project imports
print("\nChecking project imports...")

# Check for modern implementation
try:
    from src.transformer.utils.modern_data_handling import ModernField, ModernBucketIterator
    print("✓ Modern API implementation (ModernField, ModernBucketIterator) is available")
except ImportError:
    print("✗ Modern API implementation is not available or couldn't be imported")
    print("  Make sure the project structure is correct")

# Check for data_handling
try:
    from src.transformer.utils.data_handling import IWSLTDataset, USING_LEGACY
    print(f"✓ Data handling module is available")
    print(f"  Using legacy API: {USING_LEGACY}")
except ImportError:
    print("✗ Data handling module could not be imported")
    print("  Make sure the project structure is correct")

# Check for debug scripts
print("\nChecking test scripts...")
for script in ["debug_torch.py", "test_modern_implementation.py", "test_training.py"]:
    if os.path.exists(script):
        print(f"✓ {script} is available")
    else:
        print(f"✗ {script} is not found")

print("\n===== Recommended Next Steps =====")
print("1. Run the diagnostic script: python debug_torch.py")
print("2. Test the modern implementation: python test_modern_implementation.py")
print("3. Test training: python test_training.py")
print("4. Read the migration summary: migration_summary.md")

print("\nFor more information, see the README.md file.")
print("===== End of API Check =====") 