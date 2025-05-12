# Transformer Neural Machine Translation (NMT)

A PyTorch implementation of the Transformer model for Neural Machine Translation, based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Project Overview

This project implements a Transformer-based Neural Machine Translation system for English to Vietnamese translation. The implementation is based on the architecture described in the "Attention Is All You Need" paper, which introduced the Transformer model.

Key features:
- Complete implementation of the Transformer architecture
- Support for both modern and legacy torchtext APIs
- Integrated training, evaluation, and translation pipelines
- Simplified demo script for quick demonstrations

## Optimized for High-End Hardware

This implementation is optimized for high-end GPUs like the NVIDIA RTX 4080 Super. The optimizations include:

- Mixed precision training (FP16) for faster computation
- Multi-worker data loading and preprocessing
- Gradient accumulation for effective larger batch sizes
- Data preprocessing and caching to reduce CPU bottlenecks
- cuDNN benchmarking for optimized convolution operations

## Setup

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

2. Download the IWSLT'15 English-Vietnamese dataset and place it in the `data/IWSLT'15 en-vi/` directory with the following structure:

```
data/IWSLT'15 en-vi/
  train.en.txt
  train.vi.txt
  tst2012.en.txt
  tst2012.vi.txt
  tst2013.en.txt
  tst2013.vi.txt
```

## Architecture

The Transformer model consists of:

1. **Encoder**: Processes the source language (English)
   - Embedding layer (including positional encoding)
   - Multiple encoder layers, each with:
     - Multi-head self-attention mechanism
     - Position-wise feed-forward network
     - Residual connections and layer normalization

2. **Decoder**: Generates the target language (Vietnamese) 
   - Embedding layer (including positional encoding)
   - Multiple decoder layers, each with:
     - Masked multi-head self-attention
     - Multi-head attention over encoder output
     - Position-wise feed-forward network
     - Residual connections and layer normalization

3. **Final Linear Layer**: Projects decoder output to logits over the target vocabulary

### Key Components

- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence
- **Position-wise Feed-Forward Networks**: Apply transformations to each position independently
- **Positional Encoding**: Adds information about the position of tokens in the sequence
- **Residual Connections and Layer Normalization**: Help with training deep networks

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- torchtext 0.15.0+
- Additional dependencies in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo

The easiest way to see the model in action is to run the demonstration script:

```bash
python -m src.scripts.demo
```

This will:
1. Initialize a small model with reduced parameters
2. Load a fallback dataset (if IWSLT is not available)
3. Train the model for a specified number of epochs
4. Provide an interactive translation interface

### Training a Model

To train a model with default settings:

```bash
python -m src.scripts.run_model train --epochs 10
```

Additional training parameters:
- `--batch_size`: Set batch size (default: 32)
- `--learning_rate`: Set learning rate (default: 0.0005)
- `--checkpoint`: Path to save model checkpoints (default: results/checkpoints/best.pt)

### Translation

To translate text using a trained model:

```bash
python -m src.scripts.run_model translate --checkpoint results/checkpoints/best.pt --text "Hello, how are you?"
```

For interactive translation:

```bash
python -m src.scripts.run_model translate --checkpoint results/checkpoints/best.pt --interactive
```

## Project Structure

```
transformer/
├── src/
│   ├── config/
│   │   └── hyperparameters.py   # Model hyperparameters
│   ├── scripts/
│   │   ├── run_model.py         # Unified script for training/translation
│   │   └── demo.py              # Demonstration script
│   └── transformer/
│       ├── components/
│       │   ├── attention.py     # Multi-head attention implementation
│       │   ├── encoder.py       # Transformer encoder
│       │   ├── decoder.py       # Transformer decoder
│       │   └── transformer.py   # Complete Transformer model
│       └── utils/
│           ├── tokenization.py          # Tokenization utilities
│           ├── data_handling.py         # Data handling interface
│           └── modern_data_handling.py  # Modern torchtext implementation
├── results/
│   └── checkpoints/             # Saved model checkpoints
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Notes for Reviewers

This project implements a Transformer model from scratch for Neural Machine Translation. Key aspects to highlight in review:

1. **Transformer Architecture Implementation**:
   - Self-attention mechanism in `attention.py`
   - Encoder implementation in `encoder.py`
   - Decoder implementation in `decoder.py`
   - Complete model in `transformer.py`

2. **Training and Inference Pipeline**:
   - Data processing and batching
   - Training loop with validation
   - Beam search for better translations (on roadmap)

3. **Modern API Compatibility**:
   - Support for the latest torchtext API
   - Migration from legacy to modern torchtext

## Future Improvements

- Implement beam search for better translation quality
- Add support for additional language pairs
- Improve tokenization with BPE or SentencePiece
- Add model compression techniques
- Support for transfer learning from pre-trained models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementation is inspired by the paper "Attention Is All You Need" by Vaswani et al.
- Thanks to the PyTorch and torchtext teams for their excellent libraries

## Running the Model

### Training

For RTX 4080 Super / high-end GPUs:

```bash
python run_model_full.py train --batch_size 256 --grad_accum 4 --workers 4
```

This uses a batch size of 256 with 4 gradient accumulation steps (effective batch size: 1024).

### Full Training Parameters

You can adjust various parameters:

```bash
python run_model_full.py train --batch_size 256 --grad_accum 4 --workers 4 --epochs 50
```

- `--batch_size`: Physical batch size (try 192, 256, or 320 on 16GB VRAM)
- `--grad_accum`: Gradient accumulation steps (effective batch size = batch_size × grad_accum)
- `--workers`: Number of CPU worker threads for data loading (recommend 4-8 based on CPU cores)
- `--no_fp16`: Disable mixed precision training (not recommended for NVIDIA RTX GPUs)
- `--epochs`: Number of training epochs

### Translation

Translate a specific sentence:

```bash
python run_model_full.py translate --checkpoint results/checkpoints/best_model.pt --text "Hello, how are you today?"
```

Interactive translation mode:

```bash
python run_model_full.py translate --checkpoint results/checkpoints/best_model.pt --interactive
```

## Performance Recommendations

- For RTX 4080 Super (16GB): `--batch_size 256 --grad_accum 4 --workers 4`
- For RTX 4090 (24GB): `--batch_size 384 --grad_accum 4 --workers 6`
- For RTX 3080 (10GB): `--batch_size 192 --grad_accum 4 --workers 4`

## Efficient Version

For faster experimentation, you can use the efficient version which uses a smaller model and only 20% of the dataset:

```bash
python run_efficient.py train
```
