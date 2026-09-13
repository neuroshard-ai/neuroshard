import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroshard.evolution import reference_data as data


ROOT=Path(__file__).resolve().parents[2]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


driver=module('run_cooperative_learning')
serving=module('serve_compute_probe')


def test_final_test_partition_requires_explicit_scoring_mode(tmp_path):
    with pytest.raises(ValueError,match='cannot read final test'):
        driver.partition(tmp_path,{},'test')


def test_prepared_source_change_is_rejected_before_execution(tmp_path):
    plan={'name':'example'}
    data.save(tmp_path/'prepared.json',{'plan':plan,'implementation':'wrong'})
    with pytest.raises(ValueError,match='source changed'):
        driver.prepared_inputs(SimpleNamespace(home=tmp_path),plan)


def test_changed_seed_receipt_cannot_replace_committed_model_bytes(monkeypatch):
    prepared={'plan':{'model':{'repo':'expected'}},'model_snapshot':{'files':{'weights':'original'}}}
    monkeypatch.setattr(driver,'model_snapshot',lambda *a:{'files':{'weights':'changed'}})
    with pytest.raises(ValueError,match='committed preparation'):driver.verify_seed(Path('.'),prepared)


def test_candidate_binding_separates_preparation_and_experiment_arm():
    runtime={'host':'one','torch':'fixed'}
    binding=driver.binding_for({'inputs':'a'},runtime,'clean-single')
    assert binding!=driver.binding_for({'inputs':'b'},runtime,'clean-single')
    assert binding!=driver.binding_for({'inputs':'a'},runtime,'damaged-single')
    assert binding!=driver.binding_for({'inputs':'a'},runtime,'clean-pair')


def test_candidate_commitment_checks_real_git_bytes_and_model_identity(tmp_path,monkeypatch):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'config','user.name','Test'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'config','user.email','test@example.invalid'],check=True)
    prepared={'inputs':'fixed'};candidate={'candidate':'one'}
    selection=tmp_path/'selection.json'
    data.save(selection,{'prepared':data.identity(prepared),'candidates':{'clean-single':candidate}})
    subprocess.run(['git','-C',str(tmp_path),'add','selection.json'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'commit','-qm','seal'],check=True)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    driver.committed_selection(selection,prepared,'clean-single',candidate)
    with pytest.raises(ValueError,match='exact candidate'):
        driver.committed_selection(selection,prepared,'clean-single',{'candidate':'two'})
    selection.write_text(selection.read_text()+'\n')
    with pytest.raises(ValueError,match='not committed'):
        driver.committed_selection(selection,prepared,'clean-single',candidate)


def test_partition_rejects_mutated_tokenized_inputs(tmp_path):
    path=tmp_path/'inputs/dev.jsonl';path.parent.mkdir()
    row={'id':'one','input_ids':[1,2],'labels':[-100,2],'targets':1}
    path.write_text(json.dumps(row)+'\n')
    prepared={'roles':{'dev':{'ids':['one'],'sha256':data.sha256(path)}}}
    path.write_text(json.dumps({**row,'labels':[-100,3]})+'\n')
    with pytest.raises(ValueError):driver.partition(tmp_path,prepared,'dev')


def test_preparation_must_match_committed_selection(tmp_path,monkeypatch):
    path=tmp_path/'config/experiments/cooperative-learning-data-selection.json'
    path.parent.mkdir(parents=True)
    raw=b'{"inputs":"sealed"}'
    path.write_bytes(raw)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:raw)
    driver.committed_preparation({'inputs':'sealed'})
    with pytest.raises(ValueError,match='exact prepared'):
        driver.committed_preparation({'inputs':'different'})
    path.write_bytes(raw+b'\n')
    with pytest.raises(ValueError,match='exact prepared'):
        driver.committed_preparation({'inputs':'sealed'})


def test_serving_retry_reuses_answer_and_rejects_changed_task_identity():
    calls=[]
    answers=serving.Answers([{'id':'one'},{'id':'two'}],lambda r:calls.append(r['id']) or {'text':'answer'},'fixed-model')
    first=answers.answer({'request_id':'request','task_id':'one'})
    retry=answers.answer({'request_id':'request','task_id':'one'})
    assert not first['cached'] and retry['cached'] and calls==['one']
    with pytest.raises(ValueError,match='reused'):
        answers.answer({'request_id':'request','task_id':'two'})
    with pytest.raises(ValueError):answers.answer({'request_id':'new','task_id':'unknown'})
