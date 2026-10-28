"""Independent local regressions. Run: python -m pytest scripts/regression_checks.py -v.

These checks use PyTorch references and behavioral invariants. They do not
access, replace, or modify the challenge tests or grader.
"""
import itertools
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
import torch.nn.functional as F
from config import ModelConfig, InferenceConfig
from model.attention.scaled_dot_product import scaled_dot_product_attention
from model.attention.multi_head import MultiHeadAttention
from model.attention.causal_mask import create_causal_mask
from model.layers.layer_norm import LayerNorm
from model.layers.feedforward import FeedForward
from model.layers.residual import ResidualConnection
from model.embeddings import EmbeddingLayer
from model.embeddings.token_embedding import TokenEmbedding
from model.blocks import BlockStack
from model.heads.weight_tying import is_tied
from inference.engine.inference_engine import FullTransformerModel, InferenceEngine
from inference.generation import KVCache, autoregressive_generate, BeamSearch, StopCriteria
from inference.sampling import Sampler, apply_temperature, apply_top_k, apply_top_p, apply_repetition_penalty
from tokenizer import BPETokenizer, Vocabulary
from tokenizer.special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN
from serving.api.request_handler import RequestHandler
from serving.api.routes import RouteHandler
from serving.pipeline import ServingPipeline
from serving.model_loader import ModelLoader
from utils.checkpoint import save_checkpoint, load_checkpoint
from evaluation.metrics import cross_entropy_loss, token_accuracy, repetition_rate


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(421)
    torch.set_num_threads(2)


def tiny(**kwargs):
    data=dict(d_model=24,n_heads=4,n_layers=2,vocab_size=64,max_seq_len=32,dropout=0)
    data.update(kwargs)
    return ModelConfig(**data)


def close(a,b):
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('seed',range(4))
@pytest.mark.parametrize('lengths',[(1,1),(3,5),(5,3)])
@pytest.mark.parametrize('mask_kind',['none','additive','boolean','blocked'])
def test_attention_reference(seed,lengths,mask_kind):
    torch.manual_seed(seed)
    q=torch.randn(2,3,lengths[0],8,dtype=torch.float64)
    k=torch.randn(2,3,lengths[1],8,dtype=torch.float64)
    v=torch.randn(2,3,lengths[1],7,dtype=torch.float64)
    allowed=torch.rand(2,1,*lengths)>.35
    allowed[...,0]=True
    if mask_kind=='none': mask=None
    elif mask_kind=='boolean': mask=allowed
    elif mask_kind=='blocked': mask=torch.full((2,1,*lengths),float('-inf'),dtype=q.dtype)
    else: mask=torch.zeros(2,1,*lengths,dtype=q.dtype).masked_fill(~allowed,float('-inf'))
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    actual,weights=scaled_dot_product_attention(q,k,v,mask)
    close(actual,expected)
    assert torch.isfinite(weights).all()


def test_attention_gradient_reference():
    q,k,v=[torch.randn(1,2,3,4,dtype=torch.double,requires_grad=True) for _ in range(3)]
    expected=F.scaled_dot_product_attention(q,k,v)
    actual,_=scaled_dot_product_attention(q,k,v)
    a=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(q,k,v))
    for x,y in zip(a,b): close(x,y)


@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('length',[1,5])
def test_multihead_reference(bias,causal,length):
    c=tiny()
    ours=MultiHeadAttention(config=c,bias=bias).eval()
    ref=torch.nn.MultiheadAttention(c.d_model,c.n_heads,dropout=0,bias=bias,batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.q_proj.weight,ours.k_proj.weight,ours.v_proj.weight]))
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        if bias:
            ref.in_proj_bias.copy_(torch.cat([ours.q_proj.bias,ours.k_proj.bias,ours.v_proj.bias]))
            ref.out_proj.bias.copy_(ours.out_proj.bias)
    x=torch.randn(2,length,c.d_model)
    mask=create_causal_mask(length)[0,0] if causal else None
    expected,_=ref(x,x,x,attn_mask=mask)
    close(ours(x,use_causal_mask=causal),expected)


@pytest.mark.parametrize('rotary',[False,True])
@pytest.mark.parametrize('chunks',[(1,1,1,1,1,1),(3,2,1),(2,4)])
@pytest.mark.parametrize('seed',range(3))
def test_full_model_cached_chunks(rotary,chunks,seed):
    torch.manual_seed(seed)
    c=tiny(use_rotary=rotary)
    model=FullTransformerModel(c).eval()
    ids=torch.randint(0,c.vocab_size,(2,sum(chunks)))
    with torch.no_grad():
        expected=model(ids)
        cache=KVCache(c.n_layers)
        pieces=[]; start=0
        for size in chunks:
            pieces.append(model(ids[:,start:start+size],kv_cache=cache))
            start+=size
        close(torch.cat(pieces,1),expected)
        assert all(cache.get_seq_length(i)==sum(chunks) for i in range(c.n_layers))
        cache.reset()
        close(model(ids,kv_cache=cache),expected)


@pytest.mark.parametrize('shape',[(8,),[2,4],(2,3,4)])
def test_layernorm_general_shape(shape):
    x=torch.randn(3,*shape,dtype=torch.double,requires_grad=True)
    layer=LayerNorm(normalized_shape=shape).double()
    expected=F.layer_norm(x,shape,layer.weight,layer.bias,layer.eps)
    actual=layer(x)
    close(actual,expected)
    close(torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.square().sum(),x)[0])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_layernorm_precision(dtype):
    x=(torch.randn(2,3,24)*2000).to(dtype)
    layer=LayerNorm(24).to(dtype)
    expected=F.layer_norm(x,(24,),layer.weight,layer.bias,layer.eps)
    torch.testing.assert_close(layer(x),expected,atol=.02,rtol=.02)


def test_block_stack_explicit_override():
    c=tiny(d_model=32)
    stack=BlockStack(config=c,n_layers=1,d_model=16)
    assert len(stack)==1
    assert stack(torch.randn(2,3,16)).shape==(2,3,16)
    assert c.d_model==32


def test_token_padding_row():
    layer=TokenEmbedding(vocab_size=8,d_model=4,padding_idx=0)
    assert torch.equal(layer(torch.tensor([0])),torch.zeros(1,4))
    layer(torch.tensor([0,1])).sum().backward()
    assert torch.equal(layer.embedding.weight.grad[0],torch.zeros(4))


@pytest.mark.parametrize('rotary',[False,True])
def test_padding_mask_prevents_leakage(rotary):
    m=FullTransformerModel(tiny(use_rotary=rotary)).eval()
    ids=torch.tensor([[0,0,2,3,4],[0,5,6,7,8]])
    mask=torch.tensor([[0,0,1,1,1],[0,1,1,1,1]])
    changed=ids.clone(); changed[mask==0]=39
    with torch.no_grad():
        a=m(ids,attention_mask=mask)
        b=m(changed,attention_mask=mask)
    assert a.isfinite().all()
    close(a[mask.bool()],b[mask.bool()])


@pytest.mark.parametrize('seed',range(3))
def test_model_gradient_and_causality(seed):
    torch.manual_seed(seed)
    model=FullTransformerModel(tiny()).eval()
    ids=torch.randint(0,64,(2,6))
    original=model(ids)
    changed=ids.clone(); changed[:,-2:]=(changed[:,-2:]+1)%64
    close(model(changed)[:,:-2],original[:,:-2])
    original.square().mean().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(),name
    assert is_tied(model.lm_head,model.embeddings.token_embedding)


@pytest.mark.parametrize('cls',[ModelConfig,InferenceConfig])
def test_config_roundtrip(cls):
    a=cls(); b=cls.from_json(a.to_json())
    assert a.to_dict()==b.to_dict()
    if cls is ModelConfig:
        c=cls.from_dict(dict(a.to_dict(),d_model=96,n_heads=3,d_head=999,d_ffn=1))
        assert c.d_head==32 and c.d_ffn==384


@pytest.mark.parametrize('kwargs',[{'temperature':float('nan')},{'temperature':float('inf')},{'repetition_penalty':float('nan')},{'top_p':float('nan')},{'max_new_tokens':1.5},{'top_k':1.5},{'beam_width':0}])
def test_invalid_inference_config(kwargs):
    with pytest.raises(ValueError): InferenceConfig(**kwargs).validate()


@pytest.mark.parametrize('temperature',[.1,.7,1.,2.])
def test_temperature_distribution(temperature):
    logits=torch.tensor([[1.,-2.,3.,.2]])
    close(apply_temperature(logits,temperature).softmax(-1),(logits/temperature).softmax(-1))


@pytest.mark.parametrize('k',[0,1,3,8,20])
def test_topk_reference(k):
    logits=torch.randn(3,8)
    original=logits.clone()
    result=apply_top_k(logits,k)
    expected=torch.ones_like(logits,dtype=torch.bool)
    if 0<k<8:
        expected.zero_(); expected.scatter_(-1,logits.argsort(-1,descending=True)[:,:k],True)
    assert torch.equal(torch.isfinite(result),expected)
    assert torch.equal(original,logits)


@pytest.mark.parametrize('p',[0.,.2,.5,.81,1.])
@pytest.mark.parametrize('order',[(0,1,2,3),(3,1,0,2),(2,0,3,1)])
def test_nucleus_permutation_reference(p,order):
    probabilities=torch.tensor([.05,.15,.3,.5])[list(order)]
    result=apply_top_p(probabilities.log(),p)
    descending=sorted(range(4),key=lambda i:-probabilities[i].item())
    keep=[]; total=0.
    for idx in descending:
        keep.append(idx); total+=probabilities[idx].item()
        if total>p: break
    expected=torch.zeros(4,dtype=torch.bool); expected[keep]=True
    if p == 0:
        expected.fill_(True)  # Existing public API: p=0 disables filtering.
    assert torch.equal(torch.isfinite(result),expected)


def test_sampler_empirical_distribution():
    logits=torch.tensor([.1,.3,.6]).log().repeat(20000,1)
    samples=Sampler(temperature=1).sample(logits)
    frequencies=torch.bincount(samples,minlength=3)/samples.numel()
    torch.testing.assert_close(frequencies,torch.tensor([.1,.3,.6]),atol=.015,rtol=0)


@pytest.mark.parametrize('penalty',[.7,1.,1.5,3.])
def test_repetition_history_is_per_example(penalty):
    logits=torch.tensor([[4.,-3.,2.,-1.],[4.,-3.,2.,-1.]])
    histories=torch.tensor([[0,0,1],[2,3,3]])
    expected=logits.clone()
    for row,history in enumerate(histories.tolist()):
        for idx in set(history): expected[row,idx]=logits[row,idx]*penalty if logits[row,idx]<0 else logits[row,idx]/penalty
    close(apply_repetition_penalty(logits,histories,penalty),expected)
    close(logits,torch.tensor([[4.,-3.,2.,-1.],[4.,-3.,2.,-1.]]))


@pytest.mark.parametrize('history',[[-1],[4],[1.5]])
def test_invalid_history(history):
    with pytest.raises(ValueError): apply_repetition_penalty(torch.ones(4),history,2)


@pytest.mark.parametrize('text',['hello','hello world','one two three',''])
def test_tokenizer_boundaries(text):
    t=BPETokenizer()
    ids=t.encode(text)
    assert ids[0]==t.vocab.bos_token_id
    assert t.decode(ids)==text
    assert t.encode(text,add_bos=False)==ids[1:]


@pytest.mark.parametrize('text',['abc','abc abc','abc\tabc',' abc  abc '])
def test_bpe_roundtrip(text):
    t=BPETokenizer(vocab=Vocabulary({'a':2,'b':3,'c':4}),merges=[('a','b')])
    assert t.decode(t.encode(text))==text


def test_custom_vocab_is_bijective_and_isolated():
    a=Vocabulary({'alpha':0,'beta':1,BOS_TOKEN:7})
    b=Vocabulary({'gamma':0,'delta':1,BOS_TOKEN:6})
    for v in [a,b]:
        for token,idx in v._token_to_id.items(): assert v.id_to_token(idx)==token
        assert v.token_to_id(PAD_TOKEN)==v.pad_token_id
        assert v.token_to_id(BOS_TOKEN)==v.bos_token_id==v.eos_token_id
    assert a.id_to_token(0)=='alpha' and b.id_to_token(0)=='gamma'
    assert a.token_to_id('new-a')!=a.bos_token_id


def test_bpe_uses_merge_rank_even_for_existing_word():
    t=BPETokenizer(vocab=Vocabulary({'abc':10}),merges=[('b','c'),('a','b')])
    assert t.bpe('abc')==['a','bc']
    assert t._merge_word(list('ababab'),('a','b'))==['ab','ab','ab']


class SequenceModel(torch.nn.Module):
    def __init__(self,vocab=64,next_id=9,fail=False):
        super().__init__(); self.vocab=vocab; self.next_id=next_id; self.fail=fail
    def forward(self,ids):
        if self.fail: raise RuntimeError('real failure')
        assert not self.training and not torch.is_grad_enabled()
        logits=torch.full((*ids.shape,self.vocab),-10.,device=ids.device)
        logits[...,self.next_id]=10.
        return logits


@pytest.mark.parametrize('count',[0,1,5])
@pytest.mark.parametrize('cache',[False,True])
def test_generation_prefix_and_count(count,cache):
    c=tiny(); m=FullTransformerModel(c)
    x=torch.tensor([[1,2,3],[4,5,6]])
    a=autoregressive_generate(m,x,count,use_cache=cache)
    assert torch.equal(a[:,:3],x) and a.shape==(2,3+count)


def test_engine_stop_and_zero():
    c=tiny(); prompt=[1,2]
    settings=InferenceConfig(max_new_tokens=5,temperature=0,stop_token_ids=[9])
    e=InferenceEngine(c,settings,SequenceModel())
    assert e.generate_tokens(prompt)==prompt+[9]
    assert e.generate_tokens(prompt,0)==prompt
    settings.early_stopping=False
    assert e.generate_tokens(prompt)==prompt+[9]*5
    with pytest.raises(RuntimeError,match='real failure'):
        InferenceEngine(c,settings,SequenceModel(fail=True)).generate_tokens(prompt)


@pytest.mark.parametrize('beam',[1,3])
def test_engine_context_window(beam):
    c=tiny(max_seq_len=4)
    e=InferenceEngine(c,InferenceConfig(max_new_tokens=10,temperature=0,beam_width=beam),FullTransformerModel(c))
    result=e.generate_tokens([1,2,3])
    assert len(result)<=4


def test_beam_greedy_and_prompt_eos():
    m=SequenceModel()
    x=torch.tensor([[9]])
    assert BeamSearch(1,3,9).search(m,x).tolist()==[[9,9]]
    assert BeamSearch(1,0,9).search(m,x).tolist()==[[9]]


class PlainTokenizer:
    def encode(self,text): return [2,3] if text else []
    def decode(self,ids): return ' '.join(str(x) for x in ids)


class MinimalEngine:
    def generate_tokens(self,prompt_tokens,max_new_tokens,temperature):
        return list(prompt_tokens)+[9]*max_new_tokens


def test_api_preserves_minimal_engine_contract():
    response=RequestHandler(MinimalEngine(),PlainTokenizer()).handle_completion({'prompt':'hi','max_tokens':2})
    assert response['choices'][0]['text']=='9 9'
    assert response['usage']==dict(prompt_tokens=2,completion_tokens=2,total_tokens=4)


@pytest.mark.parametrize('body',[[],[1],0,False,'hello'])
def test_api_rejects_nonobject_json(body):
    status,_=RouteHandler(RequestHandler(MinimalEngine(),PlainTokenizer())).dispatch('POST','/v1/completions',body)
    assert status==400


@pytest.mark.parametrize('field,value',[('max_tokens',True),('max_tokens',-1),('temperature',float('nan')),('temperature',float('inf')),('top_k',-1),('top_p',2)])
def test_api_rejects_invalid_sampling(field,value):
    status,_=RouteHandler(RequestHandler(MinimalEngine(),PlainTokenizer())).dispatch('POST','/v1/completions',{'prompt':'hi',field:value})
    assert status==400


@pytest.mark.parametrize('pickle_format',[False,True])
def test_checkpoint_roundtrip(tmp_path,pickle_format):
    c=tiny(); m=FullTransformerModel(c).eval()
    path=tmp_path/'weights.pt'
    save_checkpoint(m.state_dict(),str(path),use_pickle=pickle_format)
    loaded=ModelLoader(str(path)).load_weights(FullTransformerModel(c),strict=True).eval()
    ids=torch.tensor([[1,2,3]])
    close(m(ids),loaded(ids))
    assert is_tied(loaded.lm_head,loaded.embeddings.token_embedding)


def test_metrics_noncontiguous_and_ignored():
    logits=torch.randn(2,4,10).transpose(0,1)
    targets=torch.randint(0,10,(2,4)).transpose(0,1); targets[0,0]=-100
    close(cross_entropy_loss(logits,targets),F.cross_entropy(logits.reshape(-1,10),targets.reshape(-1)))
    assert token_accuracy(torch.ones(2,3,dtype=torch.long),torch.full((2,3),-100))==0
    assert repetition_rate([1,2,1,2],2)==pytest.approx(1/3)


@pytest.mark.parametrize('rotary',[False,True])
@pytest.mark.parametrize('left_padding',[False,True])
def test_padded_batch_matches_separate_sequences(rotary,left_padding):
    model=FullTransformerModel(tiny(use_rotary=rotary)).eval()
    from inference.engine.batch_processor import BatchProcessor
    seqs=[[2,3,4],[5,6]]
    ids,mask=BatchProcessor().pad_sequences(seqs,side='left' if left_padding else 'right')
    with torch.no_grad():
        batch=model(ids,attention_mask=mask)
        for i,seq in enumerate(seqs):
            expected=model(torch.tensor([seq]))[0]
            close(batch[i,mask[i].bool()],expected)


@pytest.mark.parametrize('rotary',[False,True])
def test_cached_padding_mask_matches_full(rotary):
    model=FullTransformerModel(tiny(use_rotary=rotary)).eval()
    ids=torch.tensor([[0,0,2,3,4,5],[0,6,7,8,9,10]])
    mask=torch.tensor([[0,0,1,1,1,1],[0,1,1,1,1,1]])
    with torch.no_grad():
        expected=model(ids,attention_mask=mask)
        cache=KVCache(2)
        a=model(ids[:,:4],attention_mask=mask[:,:4],kv_cache=cache)
        b=model(ids[:,4:],attention_mask=mask,kv_cache=cache)
        close(torch.cat((a,b),1),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_attention_low_precision_reference(dtype):
    q,k,v=[torch.randn(2,3,4,8).to(dtype) for _ in range(3)]
    out,_=scaled_dot_product_attention(q,k,v,create_causal_mask(4))
    ref=F.scaled_dot_product_attention(q,k,v,attn_mask=create_causal_mask(4))
    torch.testing.assert_close(out,ref,atol=.02,rtol=.02)


def test_fully_blocked_attention_has_finite_gradients():
    q,k,v=[torch.randn(1,2,3,4,requires_grad=True) for _ in range(3)]
    out,w=scaled_dot_product_attention(q,k,v,torch.full((3,3),float('-inf')))
    out.sum().backward()
    for item in [q,k,v]: assert item.grad.isfinite().all() and not item.grad.any()
    assert not out.any() and not w.any()


@pytest.mark.parametrize('pickle_format',[False,True])
def test_checkpoint_map_location(tmp_path,pickle_format):
    path=tmp_path/'device.pt'
    save_checkpoint({'tensor':torch.ones(2)},str(path),use_pickle=pickle_format)
    assert load_checkpoint(str(path),map_location='meta')['tensor'].device.type=='meta'


def test_kv_cache_rejects_bad_updates_without_corruption():
    cache=KVCache(2)
    key=torch.ones(1,2,3,4)
    cache.update(0,key,key)
    with pytest.raises(ValueError): cache.update(0,torch.ones(1,2,1,4),torch.ones(2,2,1,4))
    assert cache.get_seq_length(0)==3
    with pytest.raises(ValueError): cache.update(-1,key,key)


def test_beam_matches_exhaustive_search():
    class TransitionModel(torch.nn.Module):
        def forward(self,ids):
            scores=torch.tensor([[.5,1.2,-.2],[1.5,-.1,.6],[.4,.3,1.1]])
            return scores[ids]
    model=TransitionModel()
    prompt=torch.tensor([[2]])
    outcomes=[]
    def expand(seq,score,depth):
        if depth==3 or (len(seq)>1 and seq[-1]==0):
            outcomes.append((seq,score/len(seq))); return
        probabilities=model(torch.tensor([seq]))[0,-1].log_softmax(-1)
        for idx in range(3): expand(seq+[idx],score+probabilities[idx].item(),depth+1)
    expand([2],0.,0)
    expected=max(outcomes,key=lambda entry:entry[1])[0]
    assert BeamSearch(27,3,0).search(model,prompt).tolist()==[expected]


def test_serving_tiny_model_with_custom_tokenizer():
    vocab=Vocabulary({PAD_TOKEN:0,UNK_TOKEN:1,BOS_TOKEN:2,'hello':3,'world':4})
    tokenizer=BPETokenizer(vocab=vocab)
    model=FullTransformerModel(tiny()).eval()
    pipeline=ServingPipeline(model,tokenizer,inference_config=InferenceConfig(temperature=0,max_new_tokens=2))
    assert pipeline.process('hello world',0)=='hello world'
    assert isinstance(pipeline.process('hello world',2),str)


def test_http_endpoints_and_invalid_utf8():
    import http.client,json
    from serving.api.server import TransformerServer
    from config.server_config import ServerConfig
    server=TransformerServer(ServerConfig(host='127.0.0.1',port=0),MinimalEngine(),PlainTokenizer())
    server.start(background=True)
    host,port=server._server.server_address
    def request(method,path,body=None):
        conn=http.client.HTTPConnection(host,port,timeout=5)
        try:
            conn.request(method,path,body=body,headers={'Content-Type':'application/json'})
            result=conn.getresponse()
            return result.status,json.loads(result.read())
        finally: conn.close()
    try:
        assert request('GET','/health')[0]==200
        assert request('GET','/v1/models')[0]==200
        status,data=request('POST','/v1/completions',json.dumps({'prompt':'hi','max_tokens':2}))
        assert status==200 and data['choices'][0]['text']=='9 9'
        assert request('POST','/v1/completions','{')[0]==400
        assert request('POST','/v1/completions',b'\xff')[0]==400
        assert request('GET','/missing')[0]==404
    finally:
        server.stop()



def test_custom_tokenizer_eos_stops_generation():
    vocab=Vocabulary({PAD_TOKEN:0,UNK_TOKEN:1,BOS_TOKEN:2,'hello':3})
    tokenizer=BPETokenizer(vocab=vocab)
    e=InferenceEngine(tiny(),InferenceConfig(max_new_tokens=4,temperature=0),SequenceModel(next_id=2))
    assert e.generate('hello',tokenizer=tokenizer)=='hello'
    # Explicitly verify only one EOS was generated, not four ignored tokens.
    assert e.generate_tokens([2,3],eos_token_id=2)==[2,3,2]


def test_dropout_one_and_zero_layer_model():
    c=tiny(n_layers=0,dropout=1)
    m=FullTransformerModel(c)
    e=InferenceEngine(c,InferenceConfig(max_new_tokens=2,temperature=0),m)
    assert len(e.generate_tokens([1,2]))==4
    assert not m.training


def test_api_generate_only_engine():
    class GenerateOnly:
        def generate(self,prompt,tokenizer=None,max_new_tokens=None): return prompt+' 9 9'
    response=RequestHandler(GenerateOnly(),PlainTokenizer()).handle_completion({'prompt':'hi','max_tokens':2})
    assert response['choices'][0]['text']==' 9 9'
    assert isinstance(response['usage']['completion_tokens'],int)


def test_model_can_learn_a_tiny_batch():
    torch.manual_seed(55)
    m=FullTransformerModel(tiny(n_layers=1,vocab_size=12)).train()
    inputs=torch.tensor([[1,2,3,4],[1,2,3,4]])
    targets=torch.tensor([[2,3,4,5],[2,3,4,5]])
    optimizer=torch.optim.Adam(m.parameters(),lr=.02)
    initial=cross_entropy_loss(m(inputs),targets).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss=cross_entropy_loss(m(inputs),targets)
        loss.backward(); optimizer.step()
    assert cross_entropy_loss(m(inputs),targets).item()<initial*.25



@pytest.mark.parametrize('component',['engine','pipeline','request'])
def test_compact_model_with_default_dynamic_tokenizer(component):
    c=tiny(vocab_size=64)
    model=FullTransformerModel(c).eval()
    cfg=InferenceConfig(max_new_tokens=2,temperature=0)
    tokenizer=BPETokenizer()
    engine=InferenceEngine(c,cfg,model)
    if component=='engine':
        text=engine.generate('compact model test',tokenizer,max_new_tokens=2)
        assert text.startswith('compact model test')
    elif component=='pipeline':
        text=ServingPipeline(model,tokenizer,c,cfg).process('compact model test',2)
        assert text.startswith('compact model test')
    else:
        response=RequestHandler(engine,tokenizer).handle_completion({'prompt':'compact model test','max_tokens':2,'temperature':0})
        assert isinstance(response['choices'][0]['text'],str)
        assert response['usage']['completion_tokens']<=2
    assert all(0<=idx<c.vocab_size for idx in tokenizer.encode('compact model test'))


def test_tokenize_keeps_public_word_piece_contract():
    t=BPETokenizer(vocab=Vocabulary({}),merges=[('a','b')])
    assert t.tokenize('abc abc')==['ab','c','ab','c']
    assert t.decode(t.encode('abc abc'))=='abc abc'


def test_custom_cache_capable_model_uses_incremental_inputs():
    class CacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.num_layers=1; self.lengths=[]
        def forward(self,ids,kv_cache=None):
            self.lengths.append(ids.size(1))
            if kv_cache is not None:
                states=torch.zeros(1,1,ids.size(1),4)
                kv_cache.update(0,states,states)
            logits=torch.zeros((*ids.shape,64)); logits[...,9]=10
            return logits
    model=CacheModel()
    engine=InferenceEngine(tiny(n_layers=1),InferenceConfig(max_new_tokens=3,temperature=0),model)
    assert engine.generate_tokens([2,3])==[2,3,9,9,9]
    assert model.lengths==[2,1,1]


def test_autoregressive_cache_optional_for_plain_model():
    result=autoregressive_generate(SequenceModel(),torch.tensor([[1,2]]),2,use_cache=True)
    assert result.tolist()==[[1,2,9,9]]



def test_zero_temperature_logits_remain_finite_for_greedy_callers():
    logits=torch.tensor([[1.,-3.,4.]])
    result=apply_temperature(logits,0)
    assert torch.isfinite(result).all()
    assert result.argmax(-1).item()==Sampler(temperature=0).sample(logits).item()


def test_encode_accepts_original_tokenizer_protocol():
    from tokenizer.encoding import encode
    class ExternalTokenizer:
        vocab=Vocabulary({PAD_TOKEN:0,UNK_TOKEN:1,BOS_TOKEN:2,'hello':3})
        def tokenize(self,text): return text.split()
    assert encode('hello',tokenizer=ExternalTokenizer())==[2,3]
