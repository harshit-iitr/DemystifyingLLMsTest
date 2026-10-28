"""Local regression checks; leaves the challenge's protected tests untouched.

Run: python scripts/check_regressions.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from config import ModelConfig
from model.attention import scaled_dot_product_attention
from model.layers import LayerNorm
from model.blocks import BlockStack
from model.embeddings import LearnedPositionEmbedding, SinusoidalPositionEmbedding
from evaluation.metrics import cross_entropy_loss
from serving.pipeline import ServingPipeline

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    x = torch.randn(2, 3, 4, requires_grad=True)
    ln = LayerNorm([3, 4])
    torch.testing.assert_close(ln(x), F.layer_norm(x, (3, 4), ln.weight, ln.bias, ln.eps))
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    mask = torch.full((3, 3), float('-inf'))
    out, weights = scaled_dot_product_attention(q, q, q, mask)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(weights, torch.zeros_like(weights))
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, dropout=0)
    stack = BlockStack(cfg, n_layers=1, d_model=32)
    assert stack(torch.randn(2, 3, 32)).shape == (2, 3, 32)
    assert cfg.d_model == 16
    assert len(BlockStack(cfg, n_layers=0)) == 0
    logits = torch.randn(2, 3, 5).transpose(0, 1)
    targets = torch.randint(0, 5, (2, 3)).transpose(0, 1)
    torch.testing.assert_close(cross_entropy_loss(logits, targets),
                              F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1)))
    for cls in (LearnedPositionEmbedding, SinusoidalPositionEmbedding):
        pos = cls(d_model=4, max_seq_len=8)
        for offset in (-1, 8):
            try:
                pos(seq_len=1, offset=offset)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid position accepted')
    class InvalidOutput(torch.nn.Module):
        def forward(self, ids):
            return None
    from tokenizer import BPETokenizer
    try:
        ServingPipeline(model=InvalidOutput(), tokenizer=BPETokenizer()).process('hello')
    except TypeError:
        pass
    else:
        raise AssertionError('invalid model output was silently accepted')
    print('All additional regression checks passed')

if __name__ == '__main__':
    main()
