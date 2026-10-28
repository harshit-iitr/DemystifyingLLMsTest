#!/usr/bin/env python3
"""
Weight Initialization & Checkpoint Helper.

Initializes or verifies model weights for the Broken Transformer Challenge.
By default, generates a deterministic tiny seeded checkpoint so local tests
and pipeline checks run smoothly on CPU without large external downloads.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import DEFAULT_WEIGHTS_PATH
from config.model_config import ModelConfig


def initialize_seeded_weights(output_path: str = DEFAULT_WEIGHTS_PATH, seed: int = 42):
    """
    Generate deterministic starter weights for the transformer model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.manual_seed(seed)

    config = ModelConfig()
    print(f"Initializing model weights with seed={seed} for config: {config}")

    # Build mock state dict matching GPT architecture
    state_dict = {}

    # Embeddings
    token_embedding = torch.randn(config.vocab_size, config.d_model) * 0.02
    state_dict["embeddings.token_embedding.embedding.weight"] = token_embedding

    # Blocks
    for layer_idx in range(config.n_layers):
        prefix = f"blocks.blocks.{layer_idx}"
        # Attention
        state_dict[f"{prefix}.attn.q_proj.weight"] = torch.randn(config.d_model, config.d_model) * 0.02
        state_dict[f"{prefix}.attn.k_proj.weight"] = torch.randn(config.d_model, config.d_model) * 0.02
        state_dict[f"{prefix}.attn.v_proj.weight"] = torch.randn(config.d_model, config.d_model) * 0.02
        state_dict[f"{prefix}.attn.out_proj.weight"] = torch.randn(config.d_model, config.d_model) * 0.02
        # FFN
        state_dict[f"{prefix}.ffn.linear1.weight"] = torch.randn(config.d_ffn, config.d_model) * 0.02
        state_dict[f"{prefix}.ffn.linear2.weight"] = torch.randn(config.d_model, config.d_ffn) * 0.02
        # LayerNorms
        state_dict[f"{prefix}.ln1.weight"] = torch.ones(config.d_model)
        state_dict[f"{prefix}.ln1.bias"] = torch.zeros(config.d_model)
        state_dict[f"{prefix}.ln2.weight"] = torch.ones(config.d_model)
        state_dict[f"{prefix}.ln2.bias"] = torch.zeros(config.d_model)

    state_dict["blocks.final_ln.weight"] = torch.ones(config.d_model)
    state_dict["blocks.final_ln.bias"] = torch.zeros(config.d_model)
    state_dict["lm_head.proj.weight"] = token_embedding if config.tie_weights else torch.randn(config.vocab_size, config.d_model) * 0.02

    torch.save(state_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully saved weights checkpoint to '{output_path}' ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download or initialize transformer weights.")
    parser.add_argument("--output", type=str, default=DEFAULT_WEIGHTS_PATH, help="Output path for weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic initialization")
    args = parser.parse_args()

    initialize_seeded_weights(args.output, args.seed)


if __name__ == "__main__":
    main()
