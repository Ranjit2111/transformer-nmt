# TransformerNMT

A PyTorch implementation of the Transformer model for Neural Machine Translation (English to Vietnamese) based on the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- pip package manager
- CUDA-compatible GPU (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ranjit2111/transformer-nmt.git
   cd transformer-nmt
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Spacy English language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Training a Model

To train a model from scratch:
```bash
python main.py train
```

To train with custom configuration:
```bash
python main.py train --config path/to/config.yaml
```

To resume training from a checkpoint:
```bash
python main.py train --checkpoint path/to/checkpoint.pt
```

To train for a specific number of epochs:
```bash
python main.py train --epochs 50
```

### Translating Text

To translate a single sentence:
```bash
python main.py translate --checkpoint path/to/checkpoint.pt --text "Hello, how are you?"
```

To use interactive translation mode:
```bash
python main.py translate --checkpoint path/to/checkpoint.pt --interactive
```

To adjust beam search settings for better translation:
```bash
python main.py translate --checkpoint path/to/checkpoint.pt --text "Hello, how are you?" --beam_size 10
```

## Common Issues and Solutions

### CUDA Out of Memory
If you encounter GPU memory issues, try:
- Reducing batch size in the config file
- Enabling mixed precision training (set `fp16: true` in config)
- Increasing gradient accumulation steps

### Slow Training
- Make sure you're using a GPU
- Check that mixed precision is enabled
- Increase batch size if GPU memory allows

### Missing Vocabulary
- Make sure you've run the model on the IWSLT dataset first
- Check that the checkpoint file path is correct

## Additional Information

For detailed documentation about the implementation, architecture, and code structure, please refer to [project_documentation.md](project_documentation.md).
