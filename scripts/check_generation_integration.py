"""Regression checks for serving and model adapter integration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import ModelConfig, InferenceConfig
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from inference.generation import autoregressive_generate
from serving.pipeline import ServingPipeline
torch.set_num_threads(2)
cfg=ModelConfig(d_model=16,n_heads=4,n_layers=1,vocab_size=256,dropout=0,max_seq_len=32)
m=FullTransformerModel(cfg).eval()
# A genuine model adapter accepts kwargs but does not implement cache storage.
def adapter(ids, **kwargs):
 return m(ids)
ids=torch.tensor([[65,66]])
a=autoregressive_generate(adapter,ids,3,use_cache=True)
b=autoregressive_generate(adapter,ids,3,use_cache=False)
assert torch.equal(a,b)
engine=InferenceEngine(model=m,inference_config=InferenceConfig(temperature=0,max_new_tokens=3))
assert engine.model_config is cfg
pipe=ServingPipeline(model=m,inference_config=InferenceConfig(temperature=0,max_new_tokens=3))
with torch.no_grad():
 expected = ''.join(chr(i) for i in m(ids).argmax(-1)[0].tolist())
assert pipe.process('AB') == expected
assert pipe.process('AB', max_new_tokens=3) == engine.generate('AB')
assert pipe.process('AB',max_new_tokens=0)=='AB'
print('Callable, ignored-cache, model config, and pipeline forward/decode checks passed')
