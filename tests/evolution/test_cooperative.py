import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig,LlamaForCausalLM

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import reference as engine


def model():
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=8,intermediate_size=16,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=64,tie_word_embeddings=True,attention_dropout=0.,
        attn_implementation='eager',use_cache=False))


def records(weighted=False):
    return [{'id':str(i),'input_ids':[1,3,4]+[7+i]*(i+1)+[2],
             'labels':[-100]*3+[7+i]*(i+1)+[2],'targets':i+2,
             'loss_weight':8 if weighted and i%2==0 else 1} for i in range(4)]


RECIPE={'steps':3,'warmup_steps':0,'learning_rate':.0001,'weight_decay':.01,'clip_norm':1.0}


def worker(rank,rendezvous,output,weighted):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        network=model();wrapped=DistributedDataParallel(network,broadcast_buffers=False,gradient_as_bucket_view=True)
        optimizer=engine.optimizer_for(network,RECIPE)
        for index in range(3):group.step(wrapped,optimizer,records(weighted),'cpu',RECIPE,index,rank,2)
        digest=group.parameter_digest(network)
        assert group.agree_digest(digest,2,'cpu')==[digest,digest]
        torch.save(network.state_dict(),Path(output)/f'{rank}.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('weighted',[False,True])
def test_two_processes_match_token_weighted_single_host_adam_updates(tmp_path,weighted):
    # Unequal target counts per rank expose accidental averaging of local means.
    torch.set_num_threads(1)
    reference=model();optimizer=engine.optimizer_for(reference,RECIPE)
    rows=records(weighted)
    width=max(len(r['input_ids']) for r in rows)
    tokens=torch.tensor([r['input_ids']+[0]*(width-len(r['input_ids'])) for r in rows])
    labels=torch.tensor([r['labels']+[-100]*(width-len(r['labels'])) for r in rows])
    masks=torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows])
    weights=torch.tensor([r['loss_weight'] for r in rows])
    for index in range(3):
        for param_group in optimizer.param_groups:param_group['lr']=engine.learning_rate(RECIPE,index)
        optimizer.zero_grad(set_to_none=True)
        logits=reference(input_ids=tokens,attention_mask=masks).logits[:,:-1]
        losses=torch.nn.functional.cross_entropy(logits.reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(4,-1)
        loss=(losses.sum(1)*weights).sum()/sum(r['targets']*r['loss_weight'] for r in rows)
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),RECIPE['clip_norm']);optimizer.step()
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path),weighted),nprocs=2,join=True)
    states=[torch.load(tmp_path/f'{rank}.pt',weights_only=True) for rank in range(2)]
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(states[0][key],value,rtol=1e-5,atol=1e-7)
        assert torch.equal(states[0][key],states[1][key])


def test_partition_covers_each_global_document_once():
    rows=records()
    assert [r['id'] for r in group.rank_records(rows,0,2)]==['0','2']
    assert [r['id'] for r in group.rank_records(rows,1,2)]==['1','3']
    with pytest.raises(ValueError):group.rank_records(rows[:3],0,2)
    with pytest.raises(ValueError):group.rank_records([rows[0],rows[0]],0,2)


def test_migration_ignores_hostname_only_and_digest_binds_tensor_values():
    assert group.runtime_profile({'host':'one','torch':'x'})==group.runtime_profile({'host':'two','torch':'x'})
    assert group.runtime_profile({'host':'one','torch':'x'})!=group.runtime_profile({'host':'one','torch':'y'})
    network=model();other=copy.deepcopy(network)
    assert group.parameter_digest(network)==group.parameter_digest(other)
    with torch.no_grad():next(other.parameters()).view(-1)[0]+=1
    assert group.parameter_digest(network)!=group.parameter_digest(other)
