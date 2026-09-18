import base64
import copy
from types import SimpleNamespace

import pytest

from neuroshard.evolution import semantic_questions as semantics
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork


def policy():
    positive=[1000]+[0]*383;negative=[-1000]+[0]*383
    samples=[{'id':identity([name,i]),'route':name,'features':vector}
             for name,vector in [('parent',negative),('sensor-range',positive)] for i in range(2)]
    encoder={'format':semantics.ENCODER,'files':{name:'a'*64 for name in semantics.FILES},
             'parameters':semantics.PARAMETERS,'max_tokens':512}
    return semantics.build(samples,encoder,{'sensor-range':{'route':'sensor','question':'What is the sensor range?'}})


def test_question_index_returns_access_paths_without_answers_and_preserves_rejection():
    index=semantics.Index(policy(),{'parent','sensor'})
    selected=index.select([1000]+[0]*383)
    assert selected['selected']=={'route':'sensor','question':'What is the sensor range?'}
    assert selected['distance']==0
    assert index.select([-1000]+[0]*383)['selected'] is None
    # Duplicate feature vectors resolve by the smallest committed training ID.
    ids=[r['id'] for r in policy()['rows'] if r['label']=='sensor-range']
    assert selected['training_id']==min(ids)


@pytest.mark.parametrize('attack',['answer','route','vector','length','duplicate','label','encoder','training'])
def test_unbound_or_oversized_access_material_is_rejected(attack):
    p=policy()
    if attack=='answer':p['intents']['sensor-range']['answer']='900'
    elif attack=='route':p['intents']['sensor-range']['route']='absent'
    elif attack=='vector':
        raw=bytearray(base64.b64decode(p['vectors']));raw[:2]=b'\xff\x7f';p['vectors']=base64.b64encode(raw).decode()
    elif attack=='length':p['vectors']+='AAAA'
    elif attack=='duplicate':p['rows'][1]=p['rows'][0]
    elif attack=='label':p['rows'][0]['label']='unbound'
    elif attack=='encoder':p['encoder']['parameters']=True
    elif attack=='training':p['training_root']='b'*64
    with pytest.raises(ValueError):semantics.validate(p,{'parent','sensor'})


def test_semantic_selection_changes_the_neural_input_and_records_the_original_decision():
    p=policy();service=object.__new__(PlannedGraphNetwork)
    service.config={'semantic_questions':p};service.semantic_index=semantics.Index(p,{'parent','sensor'})
    features=[1000]+[0]*383
    service.semantic_features=lambda question:{'profile':identity(p['encoder']),'input_ids':[101,25,102],'features':features}
    service.net=SimpleNamespace(rank=0,all_owners=SimpleNamespace(exchange=lambda packet:[packet,None]))
    choice={'question':'How far can the sensor measure?','decision':{'route':'parent'},'features':[1]}
    result=service.semantic_route(choice)
    assert choice['decision']['route']=='parent'
    assert result['decision']['route']=='sensor' and result['previous_decision']==choice['decision']
    assert result['semantic']['selected']['question']=='What is the sensor range?'
    assert result['semantic']['encoding']['input_ids']==[101,25,102]
    features[:]=[-1000]+[0]*383
    assert service.semantic_route(choice)['decision']==choice['decision']
    features[:]=[1000]+[0]*383
    scoped=copy.deepcopy(choice);scoped['decision']['eligible']=['parent']
    assert service.semantic_route(scoped)['semantic']['selected'] is None
    compound=copy.deepcopy(choice);compound['question']='What is the range? What is the weight?'
    assert service.semantic_route(compound)==compound


def test_complete_service_uses_canonical_neural_question_but_preserves_user_request(monkeypatch):
    from neuroshard.evolution.request_planning import ASSISTANT_POLICY
    p=policy();service=object.__new__(PlannedGraphNetwork);service.root='service';service.prefix=[]
    service.config={'semantic_questions':p,'request_policy':ASSISTANT_POLICY,'planner':{'max_tokens':64},
        'learned':{'router':{'fallback':'parent','prototypes':{'parent':[],'sensor':[]}}},
        'expert_prompts':{'sensor':{'prefix':'','suffix':'','context':'standalone'}},'general_instruction':''}
    service.semantic_index=semantics.Index(p,{'parent','sensor'});encoded=[];calls=[]
    def feature(question):
        encoded.append(question)
        return {'profile':identity(p['encoder']),'input_ids':[101,25,102],'features':[1000]+[0]*383}
    service.semantic_features=feature
    def exchange(value):
        return [value,None] if isinstance(value,dict) and 'encoding'in value else [value,value]
    service.net=SimpleNamespace(rank=0,world_size=2,verify_unchanged=lambda:None,
        all_owners=SimpleNamespace(exchange=exchange))
    monkeypatch.setattr(service,'route',lambda question:{'question':question,'decision':{'route':'parent'}})
    def call(model,messages,maximum,purpose):
        calls.append((model,messages));service.trace.append({'token_ids':[9,2]});return 'neural result'
    monkeypatch.setattr(service,'call',call)
    messages=[{'role':'user','content':'How far can the sensor measure?'}]
    result=service.answer(messages,64)
    assert encoded==[messages[0]['content']]
    assert calls==[('sensor',[{'role':'user','content':'What is the sensor range?'}])]
    assert result['request']['messages']==messages and result['plan']==[messages[0]['content']]
    assert result['answers']==[{'question':messages[0]['content'],'expert':'sensor','text':'neural result'}]
    # A contextual conversation cannot be replaced by a standalone canonical fact.
    encoded.clear();calls.clear()
    history=[{'role':'user','content':'The sensor is aboard a ship.'},{'role':'assistant','content':'Understood.'},*messages]
    service.answer(history,64)
    assert encoded==[] and calls==[('interpreter',history)]
