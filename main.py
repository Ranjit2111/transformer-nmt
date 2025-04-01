"""
Main entry point for the TransformerNMT project.
"""
import argparse
import sys
import os


def main():
    """
    Main entry point for the TransformerNMT project.
    
    This script provides a command-line interface to access the different
    functionalities of the TransformerNMT project:
    - Training the model
    - Translating text using a trained model
    """
    parser = argparse.ArgumentParser(
        description="TransformerNMT - Neural Machine Translation with Transformers",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add version argument
    parser.add_argument(
        "--version",
        action="version",
        version="TransformerNMT v1.0.0"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the TransformerNMT model"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint to resume training"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train"
    )
    
    # Translate command
    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate text using a trained model"
    )
    translate_parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    translate_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    translate_parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search"
    )
    translate_parser.add_argument(
        "--text",
        type=str,
        help="Text to translate"
    )
    translate_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive translation mode"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, print help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate script based on the command
    if args.command == "train":
        from src.scripts.train import main as train_main
        # Convert Namespace to a list of args to pass to the script
        train_args = []
        if args.config:
            train_args.extend(["--config", args.config])
        if args.checkpoint:
            train_args.extend(["--checkpoint", args.checkpoint])
        if args.epochs:
            train_args.extend(["--epochs", str(args.epochs)])
            
        # Set sys.argv for the train script
        sys.argv = [sys.argv[0]] + train_args
        train_main()
        
    elif args.command == "translate":
        from src.scripts.translate import main as translate_main
        # Convert Namespace to a list of args to pass to the script
        translate_args = ["--checkpoint", args.checkpoint]
        if args.config:
            translate_args.extend(["--config", args.config])
        if args.beam_size:
            translate_args.extend(["--beam_size", str(args.beam_size)])
        if args.text:
            translate_args.extend(["--text", args.text])
        if args.interactive:
            translate_args.append("--interactive")
            
        # Set sys.argv for the translate script
        sys.argv = [sys.argv[0]] + translate_args
        translate_main()


if __name__ == "__main__":
    main() 