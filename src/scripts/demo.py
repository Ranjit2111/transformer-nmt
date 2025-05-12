"""
Demonstration script for the Transformer NMT model.

This script provides a simplified demonstration of the Transformer NMT model
with a mini-training session that can run quickly on CPU.
"""
import torch
import time
import os
from src.transformer.utils.modern_data_handling import IWSLTDataset
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.components.transformer import TransformerNMT

MINI_HYPERPARAMETERS = {
    "hidden_dim": 128,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_heads": 2,
    "decoder_heads": 2,
    "encoder_pf_dim": 256,
    "decoder_pf_dim": 256,
    "encoder_dropout": 0.1,
    "decoder_dropout": 0.1,
    "learning_rate": 0.0001,
    "batch_size": 16,
    "max_length": 50,
    "clip": 1,
    "patience": 5
}

def train_model(dataset, model, epochs=1, save_path=None, optimizer=None):
    """Train the model for a specified number of epochs."""
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=MINI_HYPERPARAMETERS['learning_rate'])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    train_iterator, valid_iterator, _ = dataset.get_iterators()
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg
            
            optimizer.zero_grad()
            
            # Forward pass (trg is already shifted in the batch)
            output = model(src, trg[:, :-1])
            
            # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), MINI_HYPERPARAMETERS['clip'])
            
            # Update parameters
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'\rEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_iterator)} | Loss: {loss.item():.4f}', end='')
        
        train_loss /= len(train_iterator)
        print(f'\rEpoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}', end='')
        
        # Validation
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_iterator:
                src = batch.src
                trg = batch.trg
                
                # Forward pass (trg is already shifted in the batch)
                output = model(src, trg[:, :-1])
                
                # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # Calculate loss
                loss = criterion(output, trg)
                
                valid_loss += loss.item()
        
        valid_loss /= len(valid_iterator)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) // 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        print(f' | Valid Loss: {valid_loss:.4f} | Time: {epoch_mins}m {epoch_secs}s')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            if save_path:
                print(f'  Saving best model (epoch {best_epoch}) to {save_path}')
                torch.save(model.state_dict(), save_path)
    
    if save_path:
        print(f'Best model saved from epoch {best_epoch} with validation loss: {best_valid_loss:.4f}')
    
    return best_valid_loss

def translate(model, dataset, text, max_length=50, debug=False):
    """Translate a given text using the trained model."""
    model.eval()
    
    # Tokenize and convert to indices
    tokens = dataset.tokenizer.tokenize_en(text)
    
    if debug:
        print("\nDebug information:")
        print(f"Tokenized input: {tokens}")
        
    # Convert tokens to indices, use <unk> for OOV tokens
    indices = []
    unknown_tokens = []
    for token in tokens:
        if token in dataset.source_field.vocab.stoi:
            indices.append(dataset.source_field.vocab.stoi[token])
        else:
            # Use <unk> token for out-of-vocabulary words
            indices.append(dataset.source_field.vocab.stoi['<unk>'])
            unknown_tokens.append(token)
    
    if debug and unknown_tokens:
        print(f"Unknown tokens replaced with <unk>: {unknown_tokens}")
    
    # Add <eos> token
    indices.append(dataset.source_field.vocab.stoi['<eos>'])
    
    if debug:
        print(f"Token indices: {indices}")
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(model.device)
    
    # Create source mask
    src_mask = model._create_src_mask(src_tensor)
    
    # Encode source sequence
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # Initialize target sequence with <sos> token
    trg_indices = [dataset.target_field.vocab.stoi['<sos>']]
    
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(model.device)
        
        # Create target mask
        trg_mask = model._create_trg_mask(trg_tensor)
        
        # Decode encoded source sequence
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Get next token prediction
        pred_token = output.argmax(2)[:, -1].item()
        
        # Add predicted token to sequence
        trg_indices.append(pred_token)
        
        # Stop if <eos> token is predicted
        if pred_token == dataset.target_field.vocab.stoi['<eos>']:
            break
    
    # Convert indices to tokens
    trg_tokens = [dataset.target_field.vocab.itos[i] for i in trg_indices]
    
    if debug:
        print(f"Generated token indices: {trg_indices}")
        print(f"Generated tokens: {trg_tokens}")
    
    # Remove <sos> and <eos> tokens for final output
    return trg_tokens[1:-1] if trg_tokens[-1] == '<eos>' else trg_tokens[1:]

def create_model(dataset, device=None):
    """Create a TransformerNMT model with the given parameters."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with mini hyperparameters for fast training
    model = TransformerNMT(
        src_pad_idx=dataset.pad_idx,
        trg_pad_idx=dataset.pad_idx,
        trg_sos_idx=dataset.sos_idx,
        src_vocab_size=dataset.source_vocab_size,
        trg_vocab_size=dataset.target_vocab_size,
        d_model=MINI_HYPERPARAMETERS['hidden_dim'],
        n_head=MINI_HYPERPARAMETERS['encoder_heads'],
        max_seq_length=MINI_HYPERPARAMETERS['max_length'],
        ffn_hidden=MINI_HYPERPARAMETERS['encoder_pf_dim'],
        n_layers=MINI_HYPERPARAMETERS['encoder_layers'],
        dropout=MINI_HYPERPARAMETERS['encoder_dropout'],
        device=device
    )
    
    return model

def train_longer(dataset, model, epochs=10, save_path=None):
    """Train the model for more epochs to improve performance."""
    print("\nTraining model for longer to improve performance...")
    
    # Create optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=MINI_HYPERPARAMETERS['learning_rate'] * 0.5)
    
    # Train for more epochs
    train_model(dataset, model, epochs=epochs, save_path=save_path, optimizer=optimizer)

def interactive_demo():
    """Run an interactive demo of the model."""
    print("===== Transformer NMT Demo =====")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Load dataset
    print("\nInitializing tokenizer and dataset...")
    tokenizer = Tokenizer()
    dataset = IWSLTDataset(
        tokenizer=tokenizer,
        batch_size=MINI_HYPERPARAMETERS['batch_size'],
        device='cpu',
        max_length=MINI_HYPERPARAMETERS['max_length'],
        min_freq=1
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(dataset)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Check for existing checkpoint
    checkpoint_path = 'results/checkpoints/demo_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"\nLoading existing checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
        should_train = input("Do you want to train the model anyway? (y/n): ").lower() == 'y'
    else:
        should_train = True
    
    # Train model
    if should_train:
        print("\nTraining model for a quick demonstration...")
        print(f"Training with mini hyperparameters: {MINI_HYPERPARAMETERS}")
        epochs = int(input("Enter number of epochs (1-5 recommended for demo): ") or "1")
        train_model(dataset, model, epochs=epochs, save_path=checkpoint_path)
        
        # Ask if user wants to train for longer
        train_more = input("\nDo you want to train for more epochs to improve performance? (y/n): ").lower() == 'y'
        if train_more:
            more_epochs = int(input("Enter additional epochs (5-20 recommended): ") or "10")
            train_longer(dataset, model, epochs=more_epochs, save_path=checkpoint_path)
    
    # Display vocabulary information
    print(f"\nVocabulary sizes: Source = {dataset.source_vocab_size}, Target = {dataset.target_vocab_size}")
    print("Common English words in vocabulary:")
    common_words = ['hello', 'world', 'is', 'a', 'test', 'this', 'what', 'how', 'are', 'you', 'i', 'am', 'the', 'transformer']
    for word in common_words:
        if word in dataset.source_field.vocab.stoi:
            print(f"  '{word}' is in vocabulary")
        else:
            print(f"  '{word}' is NOT in vocabulary")
    
    # Interactive translation
    print("\nEntering interactive translation mode. Type 'q' to quit.")
    debug_mode = input("Enable debug mode? (y/n): ").lower() == 'y'
    
    while True:
        text = input("\nEnter English text to translate: ")
        if text.lower() == 'q':
            break
        
        translated_tokens = translate(model, dataset, text, debug=debug_mode)
        translated_text = tokenizer.detokenize_vi(translated_tokens)
        
        print(f"English: {text}")
        print(f"Vietnamese: {translated_text}")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    interactive_demo() 