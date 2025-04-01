# TransformerNMT: Project Documentation

This document provides detailed information about the implementation of the TransformerNMT project, a PyTorch implementation of the Transformer architecture for Neural Machine Translation as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Features

- Complete implementation of the Transformer architecture with:
  - Multi-head attention
  - Position-wise feed-forward networks
  - Sinusoidal positional encoding
  - Layer normalization
  - Residual connections
- Training with mixed precision support for faster training on modern GPUs
- Beam search decoding for better translation quality
- Command-line interface for training and translation
- Interactive translation mode

## Project Structure

```
/
├── main.py                  # Main entry point
├── requirements.txt         # Project dependencies
├── README.md                # Setup and usage instructions
├── project_documentation.md # Technical documentation
├── paper/                   # Research paper reference
│   └── attention is all you need.pdf
└── src/                     # Source code
    ├── config/              # Configuration handling
    │   ├── config.yaml      # Default configuration
    │   ├── hyperparameters.py # Hyperparameter class
    │   └── __init__.py
    ├── scripts/             # Command-line scripts
    │   ├── train.py         # Training script
    │   └── translate.py     # Translation script
    └── transformer/         # Transformer implementation
        ├── components/      # Transformer components
        │   ├── attention.py # Attention mechanisms
        │   ├── decoder.py   # Decoder implementation
        │   ├── embedding.py # Token and positional embeddings
        │   ├── encoder.py   # Encoder implementation
        │   ├── feed_forward.py # Feed-forward networks
        │   ├── normalization.py # Layer normalization
        │   ├── position.py  # Positional encoding
        │   ├── transformer.py # Full transformer model
        │   └── __init__.py
        ├── utils/           # Utility functions
        │   ├── data_handling.py # Dataset handling
        │   ├── tokenization.py # Tokenization utilities
        │   ├── training_utils.py # Training utilities
        │   └── __init__.py
        └── __init__.py
```

## Model Configuration

The model can be configured through a YAML configuration file. Example configuration:

```yaml
# Model parameters
model:
  d_model: 512         # Dimensionality of the model
  n_layers: 6          # Number of encoder/decoder layers
  n_heads: 8           # Number of attention heads
  ffn_hidden: 2048     # Hidden dimension in feed-forward networks
  max_seq_length: 256  # Maximum sequence length
  dropout: 0.1         # Dropout rate

# Training parameters
training:
  batch_size: 128                # Batch size
  init_lr: 1e-5                  # Initial learning rate
  factor: 0.9                    # Learning rate decay factor
  adam_eps: 5e-9                 # Adam epsilon parameter
  patience: 10                   # Patience for learning rate scheduler
  warmup_steps: 100              # Warmup steps for learning rate
  epochs: 1000                   # Maximum number of epochs
  clip: 1.0                      # Gradient clipping value
  weight_decay: 5e-4             # Weight decay for regularization
  fp16: true                     # Whether to use mixed precision training
  gradient_accumulation_steps: 4 # Steps to accumulate gradients
  pin_memory: true               # Whether to pin memory for faster data loading
  num_workers: 4                 # Number of workers for data loading

# Data parameters
data:
  dataset: "iwslt_en_vi"  # Dataset name
  min_freq: 2             # Minimum frequency for vocabulary inclusion

# Device settings
device: "cuda"  # Device to use (cuda or cpu)
```

## Component Descriptions

### Main Components

#### `src/transformer/components/transformer.py`
- `TransformerNMT`: Main Transformer model combining encoder and decoder components.

#### `src/transformer/components/encoder.py`
- `TransformerEncoder`: Full encoder module with multiple encoder blocks.
- `TransformerEncoderBlock`: Single encoder block with self-attention and feed-forward network.

#### `src/transformer/components/decoder.py`
- `TransformerDecoder`: Full decoder module with multiple decoder blocks.
- `TransformerDecoderBlock`: Single decoder block with masked self-attention, encoder-decoder attention, and feed-forward network.

#### `src/transformer/components/attention.py`
- `MultiHeadAttentionLayer`: Multi-head attention module that splits inputs into multiple heads.
- `ScaledDotProductAttention`: Core attention mechanism implementing scaled dot-product attention.

#### `src/transformer/components/embedding.py`
- `TokenEmbedding`: Module for converting token indices to embeddings.
- `TransformerEmbedding`: Combined token and positional embeddings with scaling and normalization.

#### `src/transformer/components/position.py`
- `SinusoidalPositionEmbedding`: Sinusoidal positional encoding as described in the paper.

#### `src/transformer/components/feed_forward.py`
- `PointwiseFeedForwardNetwork`: Position-wise feed-forward network with two linear layers.

#### `src/transformer/components/normalization.py`
- `TransformerLayerNorm`: Layer normalization implementation for the Transformer.

### Utilities

#### `src/transformer/utils/tokenization.py`
- `Tokenizer`: Handles tokenization for English and Vietnamese text.

#### `src/transformer/utils/data_handling.py`
- `IWSLTDataset`: Handles loading and processing of the IWSLT English-Vietnamese dataset.

#### `src/transformer/utils/training_utils.py`
- `Trainer`: Manages the training process, including mixed precision, checkpointing, and evaluation.

### Scripts

#### `src/scripts/train.py`
- Main training script that handles command-line arguments for training.

#### `src/scripts/translate.py`
- `Translator`: Handles inference for translating text, including beam search decoding.
- Main translation script that handles command-line arguments for translation.

### Configuration

#### `src/config/hyperparameters.py`
- `HyperParameters`: Class for loading and managing model hyperparameters.

#### `src/config/config.yaml`
- Default configuration file with model, training, and data parameters.

### Entry Point

#### `main.py`
- Main entry point with command-line interface for training and translation.

## Implementation Details

### Attention Mechanism

The attention mechanism is implemented as described in the paper with:
- Query, Key, and Value projections
- Parallel computation of attention heads
- Scaling by the square root of the dimension
- Masking for padding and autoregressive behavior

```python
# Simplified core of the attention mechanism
attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
attention_weights = softmax(attention_scores, dim=-1)
output = attention_weights @ value
```

### Positional Encoding

The positional encoding is implemented using sine and cosine functions of different frequencies:

```python
# Simplified positional encoding
position = torch.arange(max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

### Beam Search

For better translation quality, beam search is implemented in the `Translator` class with:
- Maintaining a list of top-k candidates at each step
- Score normalization
- Early stopping when all beams end with EOS token

### Mixed Precision Training

For faster training on modern GPUs, mixed precision training is implemented using PyTorch's `torch.cuda.amp` module:

```python
# Simplified mixed precision training
scaler = GradScaler()
with autocast():
    output = model(src, trg)
    loss = criterion(output, trg)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
scaler.step(optimizer)
scaler.update()
```

## Dataset

The IWSLT English-Vietnamese dataset is used for training and evaluation. This dataset contains around 133K sentence pairs from TED talks, making it suitable for neural machine translation tasks.

The dataset is processed using TorchText with:
- Custom tokenization for English using spaCy
- Simple whitespace tokenization for Vietnamese
- Vocabulary building with frequency threshold
- Batching of similar length sequences

## Citation

If you use this code in your research, please cite:

```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original "Attention is All You Need" paper by Google Brain team
- The PyTorch team for the excellent deep learning framework
- The TorchText team for NLP utilities 