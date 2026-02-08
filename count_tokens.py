#!/usr/bin/env python3

import argparse
from transformers import AutoTokenizer

def count_tokens_in_file(file_path: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3") -> int:
    """Count tokens in a file using the specified model's tokenizer."""
    
    # Load the tokenizer for Mistral 7B Instruct v0.3
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Tokenize and count
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    print(f"File: {file_path}")
    print(f"Characters: {len(text):,}")
    print(f"Tokens: {token_count:,}")
    print(f"Model: {model_name}")
    
    return token_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a file using Mistral 7B tokenizer")
    parser.add_argument("file_path", help="Path to the file to analyze")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", 
                       help="Model name for tokenizer (default: mistralai/Mistral-7B-Instruct-v0.3)")
    
    args = parser.parse_args()
    count_tokens_in_file(args.file_path, args.model)