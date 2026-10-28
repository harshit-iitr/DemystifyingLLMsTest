"""Check generation and probability contracts reported by CI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import InferenceConfig, ModelConfig
from inference.sampling import apply_temperature, Sampler
from inference.engine.inference_engine import FullTransformerModel
from inference.generation import autoregressive_generate

torch.set_num_threads(2)
assert InferenceConfig().max_new_tokens >= 64
scores = torch.randn(2, 100)
for temperature in (0., .5, 1., 2.):
    probs = apply_temperature(scores, temperature)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(2))
    if temperature:
        torch.testing.assert_close(probs, (scores / temperature).softmax(-1))
assert torch.equal(Sampler(temperature=0).sample(scores), scores.argmax(-1))
cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, vocab_size=100, max_seq_len=16, dropout=0)
model = FullTransformerModel(cfg).eval()
lengths = []
def cached_model(ids, kv_cache=None):
    lengths.append(ids.size(1))
    return model(ids, kv_cache=kv_cache)
ids = torch.tensor([[2, 3, 4]])
cached = autoregressive_generate(cached_model, ids, 3, use_cache=True)
assert lengths == [3, 1, 1]
uncached = autoregressive_generate(model, ids, 3, use_cache=False)
assert torch.equal(cached, uncached)
print('CI probability, capacity, and incremental-cache contracts passed')
