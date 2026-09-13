import copy

import pytest

from neuroshard.evolution import grounded_tasks as tasks


def case(family):
    return {"family":family,"query":"A100","variant":0,"rows":[
        {"id":"B200","city":"Lima","units":3,"price":7,"status":"ready","priority":8},
        {"id":"A100","city":"Oslo","units":0,"price":4,"status":"pending","priority":8},
        {"id":"C300","city":"Accra","units":2,"price":5,"status":"ready","priority":9}]}


def test_grounded_oracles_on_independently_calculated_cases():
    assert tasks.expected(case('lookup'))=={'city':'Oslo','units':0}
    missing=case('lookup');missing['query']='X999'
    assert tasks.expected(missing)=={'city':None,'units':None}
    assert tasks.expected(case('total'))=={'total':31}
    assert tasks.expected(case('filter'))=={'ids':['B200','C300']}
    assert tasks.expected(case('sort'))=={'ids':['C300','A100']}


@pytest.mark.parametrize('answer',['{"total":true}','{"total":31.0}',
    '{"total":30,"total":31}','```json\n{"total":31}\n```',
    '{"total":31,"extra":null}','{"total":NaN}','[31]'])
def test_json_correctness_rejects_wrong_types_duplicate_keys_and_extra_content(answer):
    assert not tasks.check_answer(case('total'),answer)['correct']
    assert tasks.check_answer(case('total'),' \n{"total":31}\n')['correct']


def test_control_arm_targets_are_false_without_mutating_task_inputs():
    for family in tasks.FAMILIES:
        value=case(family);before=copy.deepcopy(value)
        assert not tasks.check_answer(value,tasks.damaged_target(value))['correct']
        assert value==before


def test_generated_roles_have_disjoint_questions_and_reserved_paraphrases():
    questions={}
    for role in ['train','dev','test']:
        values=[tasks.make_case(101,role,i) for i in range(64)]
        assert values==[tasks.make_case(101,role,i) for i in range(64)]
        questions[role]={tasks.prompt(v) for v in values}
        assert len(questions[role])==64
        assert {v['variant'] for v in values}==({0,1,2,3} if role=='test' else {0,1})
    assert not questions['train'] & questions['test']
    assert not questions['dev'] & questions['test']


def test_exact_paired_accuracy_counts_discordant_pairs():
    before=[{'id':str(i),'correct':i==0} for i in range(10)]
    after=[{'id':str(i),'correct':True} for i in range(10)]
    result=tasks.paired_accuracy(before,after)
    assert result['wins']==9 and result['losses']==0
    assert result['accuracy_change']==.9 and result['one_sided_p']==1/512
    assert tasks.paired_accuracy(after,after)['one_sided_p']==1
    with pytest.raises(ValueError):tasks.paired_accuracy(before,list(reversed(after)))
