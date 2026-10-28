#!/usr/bin/env python3
"""
Main entry point for the Broken Transformer Challenge.

Run this script to inspect system behavior and diagnose failure modes:
    python run_inference.py
"""

import sys
import os

# Ensure UTF-8 stdout if supported by runtime environment. ``getattr`` keeps
# this compatible with text streams whose static type omits ``reconfigure``.
reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    try:
        reconfigure_stdout(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 65)
    print("  [CHALLENGE] The Broken Transformer Challenge")
    print("  OpenAI Student Collective x DSG | IIT Roorkee")
    print("=" * 65)
    print()

    # Step 1: Configuration
    print("[1/5] Loading system configuration...")
    try:
        from config import ModelConfig, InferenceConfig
        model_config = ModelConfig()
        inference_config = InferenceConfig()
        print(f"  [OK] ModelConfig: d_model={model_config.d_model}, n_heads={model_config.n_heads}, n_layers={model_config.n_layers}")
        print(f"  [OK] InferenceConfig: max_new_tokens={inference_config.max_new_tokens}, temp={inference_config.temperature}")
    except Exception as e:
        print(f"  [FAIL] Config initialization failed: {e}")
        return

    # Step 2: Tokenizer Check
    print("\n[2/5] Initializing tokenizer pipeline...")
    tokenizer = None
    try:
        from tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        print(f"  [OK] Tokenizer loaded: vocab_size={getattr(tokenizer, 'vocab_size', 'unknown')}")
    except Exception as e:
        print(f"  [NOTE] Tokenizer diagnostic note: {e}")

    # Step 3: Model Architecture Check
    print("\n[3/5] Instantiating transformer model layers...")
    try:
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead
        embeddings = EmbeddingLayer()
        blocks = BlockStack()
        head = LMHead()
        print("  [OK] Transformer core submodules initialized.")
    except Exception as e:
        print(f"  [NOTE] Model submodule diagnostic note: {e}")

    # Step 4: Inference Engine & Generation
    print("\n[4/5] Executing test generation...")
    prompt = "Large language models work by"
    print(f"  Prompt: \"{prompt}\"")

    try:
        from inference.engine import InferenceEngine
        engine = InferenceEngine(model_config=model_config, inference_config=inference_config)
        if hasattr(engine, "generate"):
            output = engine.generate(prompt, tokenizer=tokenizer)
            print(f"  Generated output: \"{output}\"")
        else:
            print("  [NOTE] InferenceEngine 'generate' method not yet functional.")
    except Exception as e:
        print(f"  [NOTE] Generation failure observed: {e}")

    # Step 5: Guidance
    print("\n[5/5] Diagnostic cycle complete.")
    print("-" * 65)
    print(">> Next Steps:")
    print("  1. Run the test suite:       python -m pytest tests/ -v --tb=no")
    print("  2. Direct Codex to patch:    Identify the failing module based on theory")
    print("  3. Push & check your score:  Open a Pull Request on GitHub to see live CI grading")
    print("-" * 65)


if __name__ == "__main__":
    main()
