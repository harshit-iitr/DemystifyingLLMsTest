#!/usr/bin/env python3
"""Broader self-checks (tokenizer, checkpoint, loader, serving, batching, metrics, beam, RoPE).
Run: python scripts/verify_extra.py   (writes verify_extra_report.txt next to the repo root)"""
import os, sys, math, json, tempfile, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch, torch.nn.functional as F

results = []
def run(name, fn):
    try:
        ok = bool(fn()); err = ""
    except Exception as e:
        ok = False; err = "".join(traceback.format_exception_only(type(e), e)).strip()
    results.append((name, ok, err)); print(("PASS " if ok else "FAIL ") + name + (f"  -> {err}" if err else ""))

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from config.server_config import ServerConfig
from tokenizer import BPETokenizer, Vocabulary, encode, decode
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from inference.engine.batch_processor import BatchProcessor, DynamicBatcher
from inference.generation.beam_search import BeamSearch
from inference.sampling import Sampler
from evaluation.metrics import calculate_perplexity, cross_entropy_loss, token_accuracy, repetition_rate
from evaluation.evaluate import ModelEvaluator
from model.attention.rotary_pe import RotaryEmbedding, apply_rotary_pos_emb
from model.embeddings import TokenEmbedding, EmbeddingLayer
from serving import ModelLoader, ServingPipeline
from serving.api import RequestHandler, RouteHandler
from utils.checkpoint import save_checkpoint, load_checkpoint, inspect_checkpoint
from utils.tensor_utils import pad_sequence, to_device, ensure_tensor

torch.manual_seed(0)
cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, vocab_size=50257, max_seq_len=64, dropout=0.0)

# ---- config
run("config json round-trip", lambda: ModelConfig.from_json(ModelConfig().to_json()) == ModelConfig())
run("inference config validates defaults", lambda: InferenceConfig().validate() is None and InferenceConfig().max_new_tokens > 0)
run("server config round-trip", lambda: ServerConfig.from_json(ServerConfig().to_json()) == ServerConfig())

# ---- tokenizer
def tok_roundtrip():
    t = BPETokenizer(); ids = t.encode("the quick brown fox")
    return ids[0] == 50256 and t.decode(ids) == "the quick brown fox"
run("tokenizer encode/decode round-trip with BOS", tok_roundtrip)
run("tokenizer add_bos=False has no BOS", lambda: BPETokenizer().encode("a b", add_bos=False)[0] != 50256)
run("same word -> same id", lambda: (lambda t: t.encode("a a", add_bos=False))(BPETokenizer())[0] == BPETokenizer().encode("a a", add_bos=False)[1])
run("vocab pad/unk/eos ids", lambda: (lambda v: v.pad_token_id == 0 and v.unk_token_id == 1 and v.eos_token_id == 50256)(Vocabulary()))
def bpe_ranked():
    t = BPETokenizer(merges=[("l", "o"), ("lo", "w"), ("e", "r")])
    return t.bpe("lower") == ["low", "er"]
run("BPE applies merges by rank and merges every occurrence", bpe_ranked)
run("BPE merges repeated pairs", lambda: BPETokenizer(merges=[("a", "a")]).bpe("aaaa") == ["aa", "aa"])

# ---- embeddings
run("embedding rejects out-of-range ids", lambda: _raises(lambda: TokenEmbedding(10, 4)(torch.tensor([[11]])), IndexError))
def _raises(fn, exc):
    try: fn()
    except exc: return True
    return False
run("embedding layer shape", lambda: EmbeddingLayer(config=cfg)(torch.randint(0, 100, (2, 7))).shape == (2, 7, 32))

# ---- RoPE
def rope_norm():
    r = RotaryEmbedding(16, 32); q = torch.randn(1, 2, 8, 16); k = torch.randn(1, 2, 8, 16)
    cos, sin = r(q, 8); q2, _ = apply_rotary_pos_emb(q, k, cos, sin)
    return torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-4)
run("RoPE preserves vector norm", rope_norm)

# ---- checkpoint + loader
model = FullTransformerModel(cfg).eval()
def ckpt_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        for use_pickle in (True, False):
            p = os.path.join(d, f"c{use_pickle}.pt"); save_checkpoint(model.state_dict(), p, use_pickle=use_pickle)
            sd = load_checkpoint(p)
            if set(sd) != set(model.state_dict()): return False
        return len(inspect_checkpoint(p)) == len(model.state_dict())
run("checkpoint save/load/inspect (pickle and torch)", ckpt_roundtrip)
def loader_strict():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.pt"); torch.save(model.state_dict(), p)
        fresh = FullTransformerModel(cfg).eval()
        ModelLoader(p).load_weights(fresh, strict=True)
        x = torch.randint(0, 100, (1, 5))
        with torch.no_grad(): return torch.allclose(model(x), fresh(x))
run("ModelLoader strict load reproduces outputs", loader_strict)
run("ModelLoader default path uses .pt", lambda: ModelLoader().weights_path.endswith("model_weights.pt"))
run("ModelLoader missing file raises", lambda: _raises(lambda: ModelLoader("nope.pt").load_weights(model), FileNotFoundError))

# ---- engine
def engine_seeded():
    e = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=6, stop_token_ids=[]), model=model)
    return e.generate_tokens([1, 2, 3]) == e.generate_tokens([1, 2, 3])
run("engine deterministic under a fixed seed", engine_seeded)
def engine_topk1():
    e = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=5, temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0, stop_token_ids=[]), model=model)
    g = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=5, temperature=0.0, repetition_penalty=1.0, stop_token_ids=[]), model=model)
    return e.generate_tokens([4, 5]) == g.generate_tokens([4, 5])
run("top_k=1 sampling == greedy", engine_topk1)
def engine_stop():
    e = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=10, temperature=0.0, repetition_penalty=1.0), model=model)
    first = e.generate_tokens([1, 2], max_new_tokens=1)[-1]
    e.inference_config.stop_token_ids = [first]
    return len(e.generate_tokens([1, 2], max_new_tokens=10)) == 3
run("engine stops on stop token", engine_stop)
run("engine eval mode", lambda: (lambda e: (e.generate_tokens([1], max_new_tokens=1), not e.model.training)[1])(InferenceEngine(model_config=cfg, model=FullTransformerModel(cfg))))
run("max_new_tokens=0 returns prompt", lambda: InferenceEngine(model_config=cfg, model=model).generate_tokens([1, 2], max_new_tokens=0) == [1, 2])
run("sampler: repetition penalty lowers repeated token", lambda: Sampler(temperature=0.0, repetition_penalty=5.0).sample(torch.tensor([[2.0, 1.9]]), generated_tokens=[0]).item() == 1)

# ---- beam search
def beam():
    out = BeamSearch(beam_width=2, max_length=3).search(model, torch.tensor([[1, 2]]))
    return out.shape == (1, 5) and torch.equal(out[:, :2], torch.tensor([[1, 2]]))
run("beam search extends prompt", beam)

# ---- batching
def batching():
    bp = BatchProcessor(pad_id=0)
    ids, mask = bp.pad_sequences([[1, 2, 3], [4]], side="left")
    ids2, mask2 = bp.pad_sequences([[1, 2, 3], [4]], side="right")
    un = bp.unpad_sequences(ids, mask)
    return ids.tolist() == [[1, 2, 3], [0, 0, 4]] and mask2.tolist() == [[1, 1, 1], [1, 0, 0]] and un[1].tolist() == [4]
run("batch pad/unpad left+right", batching)
def dyn():
    b = DynamicBatcher(max_batch_size=2, max_wait_ms=10_000)
    b.enqueue("a", [1]); b.enqueue("b", [1, 2])
    r = b.ready_batch()
    return r is not None and r[0] == ["a", "b"] and r[1].shape == (2, 2)
run("dynamic batcher releases full batch", dyn)
run("pad_sequence left/right", lambda: pad_sequence([torch.tensor([1]), torch.tensor([1, 2])], side="left").tolist() == [[0, 1], [1, 2]])
run("to_device keeps requested device", lambda: to_device(torch.zeros(1), "cpu").device.type == "cpu")
run("ensure_tensor dtype", lambda: ensure_tensor([1, 2], dtype=torch.float32).dtype == torch.float32)

# ---- metrics / evaluator
lg = torch.randn(2, 4, 10); tg = torch.randint(0, 10, (2, 4))
run("cross entropy matches torch", lambda: torch.allclose(cross_entropy_loss(lg, tg), F.cross_entropy(lg.view(-1, 10), tg.view(-1))))
run("perplexity = exp(loss)", lambda: abs(calculate_perplexity(1.0) - math.e) < 1e-6)
run("token accuracy perfect", lambda: token_accuracy(F.one_hot(tg, 10).float(), tg) == 1.0)
run("repetition rate", lambda: repetition_rate([1, 2, 3, 1, 2, 3], n=3) == 0.25)
run("evaluator batch keys", lambda: set(ModelEvaluator(model).evaluate_batch(torch.randint(0, 50, (1, 6)), torch.randint(0, 50, (1, 6)))) == {"loss", "perplexity", "token_accuracy"})

# ---- serving
tk = BPETokenizer()
eng = InferenceEngine(model_config=cfg, inference_config=InferenceConfig(max_new_tokens=4, temperature=0.0), model=model)
rh = RouteHandler(RequestHandler(engine=eng, tokenizer=tk))
run("GET /health", lambda: rh.dispatch("GET", "/health")[0] == 200)
run("GET /v1/models", lambda: rh.dispatch("GET", "/v1/models")[1]["data"][0]["id"])
run("unknown route 404", lambda: rh.dispatch("GET", "/nope")[0] == 404)
def completion():
    code, body = rh.dispatch("POST", "/v1/completions", {"prompt": "hello world", "max_tokens": 3, "temperature": 0.0})
    return code == 200 and isinstance(body["choices"][0]["text"], str) and body["usage"]["completion_tokens"] == 3 and body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 3
run("POST /v1/completions returns text + usage", completion)
def pipeline():
    out = ServingPipeline(model=model, tokenizer=tk, config=cfg).process("hello world")
    return isinstance(out, str)
run("ServingPipeline.process returns str", pipeline)

bad = [r for r in results if not r[1]]
report = "\n".join(("PASS " if ok else "FAIL ") + n + (f"  -> {e}" if e else "") for n, ok, e in results) + f"\n\n{len(results)-len(bad)}/{len(results)} checks passed\n"
open(os.path.join(ROOT, "verify_extra_report.txt"), "w", encoding="utf-8").write(report)
print(f"\n{len(results)-len(bad)}/{len(results)} checks passed")
sys.exit(1 if bad else 0)
