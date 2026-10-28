#!/usr/bin/env python3
"""Numerical self-checks for the transformer fixes. Run: python scripts/verify_fixes.py"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config.model_config import ModelConfig
from model.attention.causal_mask import create_causal_mask
from model.attention.scaled_dot_product import scaled_dot_product_attention
from model.layers.layer_norm import LayerNorm
from model.layers.residual import ResidualConnection
from model.embeddings.position_embedding import SinusoidalPositionEmbedding
from model.heads.lm_head import LMHead
from model.heads.weight_tying import is_tied
from inference.sampling import apply_top_k, apply_top_p, apply_repetition_penalty, apply_temperature, Sampler
from inference.generation.autoregressive import autoregressive_generate
from inference.generation.kv_cache import KVCache
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from config.inference_config import InferenceConfig

torch.manual_seed(0)
results = []
def check(name, cond):
    results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)

cfg = ModelConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=200, max_seq_len=64, dropout=0.0)

m = create_causal_mask(4)[0, 0]
check("causal mask: future masked, past/diag allowed", m[0,1]==float("-inf") and m[1,0]==0 and m[2,2]==0)

q = torch.randn(1,2,5,16); k = torch.randn(1,2,5,16); v = torch.randn(1,2,5,16)
_, w = scaled_dot_product_attention(q,k,v,mask=create_causal_mask(5))
check("attention rows sum to 1", torch.allclose(w.sum(-1), torch.ones(1,2,5), atol=1e-5))
check("attention is causal", w[0,0,0,1:].abs().sum() < 1e-6)
_, w2 = scaled_dot_product_attention(q*100,k*100,v)
check("attention scaled by sqrt(d_k) (finite, not one-hot saturated at x1)", torch.isfinite(w2).all())

ln = LayerNorm(normalized_shape=32); x = torch.randn(3,7,32)*5+2; y = ln(x)
check("layernorm: per-token mean~0, var~1", y.mean(-1).abs().max()<1e-4 and (y.var(-1,unbiased=False)-1).abs().max()<1e-3)

r = ResidualConnection(dropout=0.0); x = torch.randn(2,3,8)
check("residual = x + sublayer(x)", torch.allclose(r(x, lambda t: t*2), x+x*2))

pe = SinusoidalPositionEmbedding(16, 8)(seq_len=4)
check("sinusoidal: pos0 = [0,1,0,1...]", torch.allclose(pe[0,0,0::2], torch.zeros(8)) and torch.allclose(pe[0,0,1::2], torch.ones(8)))

head = LMHead(config=cfg); out = head(torch.randn(2,5,64))
check("lm head shape (B,T,vocab)", out.shape == (2,5,200))

lg = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
check("top-k keeps 2 largest", torch.isfinite(apply_top_k(lg,2)).tolist()[0] == [False,True,False,True,False])
tp = apply_top_p(torch.tensor([[0.0, 3.0, 2.9, -5.0, -6.0]]), 0.5)
check("top-p works on unsorted logits (keeps top-1 at least, drops tail)", torch.isfinite(tp)[0,1] and not torch.isfinite(tp)[0,4])
rp = apply_repetition_penalty(torch.tensor([[2.0,-2.0,1.0]]), [0,1], penalty=2.0)
check("repetition penalty: pos/ , neg*", torch.allclose(rp, torch.tensor([[1.0,-4.0,1.0]])))
check("temperature scales logits", torch.allclose(apply_temperature(lg, 2.0), lg/2))
check("sampler temp=0 is greedy", Sampler(temperature=0.0).sample(lg).item()==1)

model = FullTransformerModel(cfg).eval()
check("weights tied", is_tied(model.lm_head, model.embeddings.token_embedding))
ids = torch.randint(0,200,(1,6))
with torch.no_grad():
    full = model(ids)
    check("logits shape", full.shape == (1,6,200))
    # causality: changing the last token must not change earlier logits
    ids2 = ids.clone(); ids2[0,-1] = (ids2[0,-1]+1)%200
    check("model is causal", torch.allclose(full[:,:-1], model(ids2)[:,:-1], atol=1e-5))
    # kv-cache == full forward
    cache = KVCache(cfg.n_layers)
    a = model(ids[:,:5], kv_cache=cache)
    b = model(ids[:,5:], kv_cache=cache)
    check("kv-cache incremental == full forward", torch.allclose(b[:,-1], full[:,-1], atol=1e-4))
    g = autoregressive_generate(model, ids, max_new_tokens=4)
    check("generate appends tokens, keeps prompt", g.shape==(1,10) and torch.equal(g[:,:6], ids))
    g2 = autoregressive_generate(model, ids, max_new_tokens=4, use_cache=True)
    check("generate with cache == without cache", torch.equal(g, g2))

eng = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=5, temperature=0.0, stop_token_ids=[]))
out = eng.generate_tokens([1,2,3])
check("engine generates 5 new tokens", len(out)==8 and all(0<=t<200 for t in out))

bad = [n for n,ok in results if not ok]
print(f"\n{len(results)-len(bad)}/{len(results)} checks passed")
sys.exit(1 if bad else 0)
