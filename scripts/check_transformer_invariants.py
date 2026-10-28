"""Run focused invariants for the reported transformer benchmark failures."""
from types import SimpleNamespace
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.inference_config import InferenceConfig
from evaluation.metrics import compute_perplexity
from inference.engine.batch_processor import BatchProcessor
from inference.generation.autoregressive import autoregressive_generate
from model.blocks.transformer_block import TransformerBlock
from model.embeddings.embedding_layer import EmbeddingLayer
from model.heads.lm_head import LMHead
from model.heads.weight_tying import is_tied, tie_weights
from serving.pipeline import ServingPipeline


def check_stage_09():
    config = SimpleNamespace(
        d_model=4, n_heads=2, n_layers=1, vocab_size=8, max_seq_len=8,
        ffn_multiplier=4, dropout=0.0, layer_norm_eps=1e-5, use_rotary=False,
    )
    block = TransformerBlock(config=config)

    class Add(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x):
            return x + self.value

    class Attention(nn.Module):
        def forward(self, x, mask=None):
            assert mask is not None
            return 2 * x

    class MLP(nn.Module):
        def forward(self, x):
            return 3 * x

    block.ln1 = Add(1)
    block.attn = Attention()
    block.ln2 = nn.Identity()
    block.mlp = MLP()
    block.dropout = nn.Identity()
    x = torch.ones(1, 2, 4)
    actual = block(x, mask=torch.ones(1, 2))
    expected = x + 2 * (x + 1) + 3 * (x + 2 * (x + 1))
    assert torch.equal(actual, expected), "Pre-LN residual order mismatch"


def check_stage_33():
    logits = torch.tensor([[[0., 8., 0.], [0., 0., 8.], [8., 0., 0.]]])
    labels = torch.tensor([[2, 1, 0]])
    actual = compute_perplexity(logits, labels, pad_token_id=0)
    expected_loss = torch.nn.functional.cross_entropy(logits[:, 0, :], labels[:, 1])
    assert abs(actual - torch.exp(expected_loss).item()) < 1e-6, "Shift or pad masking mismatch"


def check_integration_01():
    config = SimpleNamespace(
        d_model=4, vocab_size=8, max_seq_len=8, use_rotary=True,
        dropout=0.0, scale_embeddings=True,
    )
    embeddings = EmbeddingLayer(config=config)
    ids = torch.tensor([[1, 2]])
    raw = embeddings.token_embedding.embedding(ids)
    assert torch.allclose(embeddings.token_embedding(ids), raw * 2), "Embedding scale not propagated"
    head = LMHead(config=config)
    tie_weights(head, embeddings.token_embedding)
    assert is_tied(head, embeddings.token_embedding), "Embedding/head parameters are not tied by identity"


def check_integration_03():
    class CacheProbe(nn.Module):
        num_layers = 1

        def __init__(self):
            super().__init__()
            self.positions = []

        def forward(self, input_ids, kv_cache=None, position_ids=None, attention_mask=None):
            self.positions.append(position_ids.clone())
            key = torch.zeros(input_ids.size(0), 1, input_ids.size(1), 2)
            kv_cache.update(0, key, key)
            return torch.zeros(input_ids.size(0), input_ids.size(1), 6)

    model = CacheProbe()
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    autoregressive_generate(
        model, torch.tensor([[0, 4, 5], [2, 3, 4]]),
        max_new_tokens=2, use_cache=True, attention_mask=mask,
    )
    assert model.positions[0].tolist() == [[0, 0, 1], [0, 1, 2]], "Prefill positions mismatch"
    assert model.positions[1].tolist() == [[2], [3]], "Cached decode positions mismatch"


def check_integration_05():
    padded, mask = BatchProcessor(pad_id=0).pad_sequences([[4, 5], [6]])
    assert padded.tolist() == [[4, 5], [0, 6]], "Prompts are not left padded"
    assert mask.tolist() == [[1, 1], [0, 1]], "Padding mask is incorrect"

    class Tokenizer:
        def encode(self, text):
            return [4, 5]

        def decode(self, ids):
            return str(ids)

    class ServingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.masks = []

        def forward(self, input_ids, attention_mask=None):
            assert attention_mask is not None, "Serving did not pass attention mask"
            self.masks.append(attention_mask.clone())
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8)
            logits[:, :, 2] = 10
            return logits

    model = ServingModel()
    config = InferenceConfig(
        max_new_tokens=2, temperature=0, top_k=0, top_p=1,
        stop_token_ids=[], seed=None,
    )
    ServingPipeline(model=model, tokenizer=Tokenizer(), inference_config=config).process("x")
    assert [m.tolist() for m in model.masks] == [[[1, 1]], [[1, 1, 1]]], "Serving mask was not extended"


def main():
    checks = [
        check_stage_09,
        check_stage_33,
        check_integration_01,
        check_integration_03,
        check_integration_05,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Local invariant checks: {len(checks)} / {len(checks)}")


if __name__ == "__main__":
    main()
