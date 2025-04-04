"""
Script to move redundant files to the redundant directory.

This script is used to clean up the codebase by moving
redundant files to dedicated folders, making it easier to
understand the essential parts of the codebase.
"""
import os
import shutil
from pathlib import Path

# Ensure the redundant directories exist
redundant_dir = Path("redundant")
redundant_scripts = redundant_dir / "scripts"
redundant_utils = redundant_dir / "utils"
redundant_tests = redundant_dir / "tests"

for directory in [redundant_dir, redundant_scripts, redundant_utils, redundant_tests]:
    directory.mkdir(exist_ok=True)

# List of redundant script files to move
redundant_script_files = [
    Path("src/scripts/train.py"),
    Path("src/scripts/translate.py"),
    Path("main.py"),
    Path("api_check.py"),
    Path("debug_torch.py"),
]

# List of redundant utility files to move
redundant_util_files = [
    Path("src/transformer/utils/hyperparameters.py"),
    Path("migration_plan.md"),
    Path("migration_summary.md"),
    Path("MAINTENANCE.md"),
]

# List of redundant test files to move
redundant_test_files = [
    Path("test_modern_implementation.py"),
    Path("test_training.py"),
]

def move_file(src, dest_dir):
    """Move a file to the destination directory."""
    if not src.exists():
        print(f"File not found: {src}")
        return False
    
    dest = dest_dir / src.name
    if dest.exists():
        print(f"Destination already exists: {dest}")
        return False
    
    try:
        # Create parent directories if needed
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file first
        shutil.copy2(src, dest)
        print(f"Copied: {src} -> {dest}")
        
        # Only remove the original after successful copy
        os.remove(src)
        print(f"Removed original: {src}")
        return True
    except Exception as e:
        print(f"Error moving {src} to {dest}: {e}")
        return False

def main():
    """Move all redundant files to their appropriate directories."""
    print("Moving redundant script files...")
    for file in redundant_script_files:
        move_file(file, redundant_scripts)
    
    print("\nMoving redundant utility files...")
    for file in redundant_util_files:
        move_file(file, redundant_utils)
    
    print("\nMoving redundant test files...")
    for file in redundant_test_files:
        move_file(file, redundant_tests)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 