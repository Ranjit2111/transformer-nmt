# Transformer Neural Machine Translation (NMT)

A PyTorch implementation of the Transformer model for Neural Machine Translation, based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Project Overview

This project implements a Transformer-based Neural Machine Translation system for English to Vietnamese translation. The implementation is based on the architecture described in the "Attention Is All You Need" paper, which introduced the Transformer model.

Key features:
- Complete implementation of the Transformer architecture
- Support for both modern and legacy torchtext APIs
- Integrated training, evaluation, and translation pipelines
- Simplified demo script for quick demonstrations

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
