# Transformer Implementation for Neural Machine Translation

This repository contains an implementation of the Transformer architecture for Neural Machine Translation, specifically focusing on English-Vietnamese translation using the IWSLT dataset.

## Project Overview

The Transformer model is based on the architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. It uses self-attention mechanisms instead of recurrent or convolutional neural networks to capture dependencies between input and output sequences.

## Prerequisites

- Python 3.8+ (3.10 recommended)
- NVIDIA GPU with CUDA support (for faster training)
- Windows 10/11 or Linux

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/transformer.git
cd transformer
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac
```

### 3. Install CUDA Toolkit (for GPU acceleration)

If you have an NVIDIA GPU and want to enable GPU acceleration (highly recommended for training):

#### a. Check your GPU model and compatibility:
```bash
# Windows PowerShell
powershell -Command "Get-WmiObject -Class Win32_VideoController | Select-Object Name, DriverVersion"
```

#### b. Download and Install CUDA Toolkit:
1. Go to [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your operating system, architecture, and version
3. Download and run the installer (CUDA 11.8 recommended for PyTorch 2.0.0)
4. Choose "Express Installation"
5. Restart your computer after installation

#### c. Verify CUDA Installation:
```bash
nvcc --version
```

#### d. Install cuDNN:
1. Go to [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires free NVIDIA developer account)
2. Download cuDNN for your CUDA version
3. Extract and copy files to your CUDA installation:
   - Copy `bin/*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
   - Copy `include/*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`
   - Copy `lib/*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

### 4. Install Required Dependencies

First, install the core dependencies:

```bash
pip install numpy==1.24.3
```

Then install PyTorch and torchtext with the correct versions:

#### For CPU-only:
```bash
pip install torch==2.0.0 torchtext==0.15.0
```

#### For GPU acceleration:
```bash
pip install torch==2.0.0 torchtext==0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

Finally, install the remaining dependencies:
```bash
pip install spacy>=3.0.0 tqdm>=4.48.0 pyyaml>=5.4.0 matplotlib>=3.3.0 tensorboard>=2.4.0
```

### 5. Download Spacy Language Models

```bash
python -m spacy download en_core_web_sm
```

### 6. Verify Installation

Run the diagnostic script to verify your installation:

```bash
python debug_torch.py
```

This should show that PyTorch and torchtext are installed correctly. If torchtext.legacy is not available, you may need to reinstall torchtext:

```bash
pip uninstall -y torchtext
pip install torchtext==0.15.0
```

### 7. Verify GPU Detection

```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

This should print `GPU available: True` and your GPU model name if everything is set up correctly.

## Modern API Support

This project now supports both the legacy (`torchtext.legacy`) and modern torchtext API (v0.15.0+). The migration was necessary because newer versions of torchtext (≥0.12) have removed the legacy API components like `Field` and `BucketIterator`.

### How It Works

- The codebase automatically detects which API is available and routes to the appropriate implementation
- Legacy API is used when available for backward compatibility
- Modern implementation is used automatically when the legacy API is not available
- All training and inference pipelines work with both implementations

### Testing the Modern Implementation

To verify that the modern implementation works correctly:

```bash
# Test the modern implementation components
python test_modern_implementation.py

# Test training and inference with the modern implementation
python test_training.py
```

For more details about the migration, see the [Migration Summary](migration_summary.md).

## Running the Project

### Using the Unified Script

The easiest way to run the project is using the unified script:

```bash
# Training
python -m src.scripts.run_model train --epochs 10

# Evaluation
python -m src.scripts.run_model evaluate --checkpoint results/checkpoints/best.pt

# Translation
python -m src.scripts.run_model translate --checkpoint results/checkpoints/best.pt --text "Hello, how are you?"

# Interactive translation
python -m src.scripts.run_model translate --checkpoint results/checkpoints/best.pt --interactive
```

The unified script works with both the legacy and modern torchtext APIs.

### Legacy Scripts

Alternatively, you can use the individual scripts:

#### Training the Model

To train the model with default parameters:

```bash
python -m src.scripts.train --epochs 10
```

Additional training parameters can be adjusted in `src/config/config.yaml` or specified as command-line arguments:

```bash
python -m src.scripts.train --epochs 20 --batch_size 64 --learning_rate 0.0001
```

#### Evaluating the Model

To evaluate a trained model on the test set:

```bash
python -m src.scripts.translate --checkpoint results/checkpoints/best.pt --evaluate
```

#### Translating Text

To translate custom text using a trained model:

```bash
python -m src.scripts.translate --checkpoint results/checkpoints/best.pt --text "Your English text here"
```

## Project Structure

- `main.py`: Entry point for training, evaluation, and inference
- `src/transformer/`: Core model implementation
  - `layers/`: Transformer building blocks (attention, encoders, decoders)
  - `utils/`: Utility functions for data processing and visualization
- `src/config/`: Configuration files
- `src/scripts/`: Training, evaluation, and inference scripts

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy compatibility errors:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Make sure to install NumPy 1.24.3 as specified in requirements.txt:

```bash
pip install numpy==1.24.3
```

### CUDA Not Detected

If PyTorch doesn't detect your GPU:

1. Verify CUDA installation: `nvcc --version`
2. Check GPU drivers are up to date
3. Ensure you installed the CUDA-enabled version of PyTorch
4. Restart your computer

### torchtext.legacy Not Available

If your code fails with:
```
ImportError: cannot import name 'Field' from 'torchtext.legacy.data'
```

Try reinstalling torchtext with the exact version needed:

```bash
pip uninstall -y torchtext
pip install torchtext==0.15.0
```

If the issue persists, check if you're using pip's cache, which might have a corrupted package:

```bash
pip cache purge
pip install torchtext==0.15.0
```

## License

[MIT License](LICENSE)

## Acknowledgments

- Paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- IWSLT Dataset: [International Workshop on Spoken Language Translation](https://iwslt.org/)

## Project Maintenance

For information about maintaining the project, including details about:
- Test files and which ones can be removed
- Configuration and hyperparameters management
- Modern vs legacy API compatibility

Please see the [Maintenance Guide](MAINTENANCE.md).
