import json
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_documents_demonstration as demo
from neuroshard.evolution import assistant_general_l4 as l4
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_growth_baseline import cloud_module


def test_every_training_case_has_a_verified_demonstration_and_none_other():
    policy = l4.policies()['documents']
    rows = demo.all_demonstrations(policy)
    assert len(rows) == len(docs.cases('train')) == 132 and all(row['passed'] for _, row in rows)
    for split in ('development', 'confirmation', 'confirmation2'):
        with pytest.raises(ValueError):
            demo.texts(docs.cases(split)[0])
    case, row = rows[0]
    item = demo.trajectory(case, row, policy)
    assert item['messages'][0]['role'] == 'system' and item['trainable'] == [m['role'] == 'assistant'
                                                                            for m in item['messages']]


def test_not_found_demonstrations_read_the_outbound_itinerary_and_report():
    case = next(c for c in docs.cases('train') if c['family'] == 'not-found-return')
    texts = demo.texts(case)
    assert 'read_document' in texts[1] and 'report_not_found' in texts[2]
    assert 'submit_answer' not in ''.join(texts)


def test_spec_uses_a2_settings_with_the_declared_steps_and_mixture():
    settings, declared = l4.spec(), l4.declared()
    assert settings['steps'] == declared['training']['steps'] and settings['mixture'] == declared['training']['mixture']
    assert sum(settings['mixture'].values()) == settings['gradient_accumulation']


def test_a_tiny_l4_trains_saves_and_reloads(tmp_path):
    torch = pytest.importorskip('torch')
    from test_assistant_experience import TEMPLATE, tiny_model
    from test_granite_tokenizer import granite_like, load_tiny
    from neuroshard.evolution import assistant_documents as documents
    from neuroshard.evolution import assistant_experience_train as trainer

    directory = granite_like(tmp_path)
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    policy = l4.policies()['documents']
    case, row = demo.all_demonstrations(policy)[0]
    sequence = trainer.encode(tokenizer, demo.trajectory(case, row, policy), documents.TOOLS)
    settings = {**l4.spec(), 'layers': [0, 1], 'steps': 2, 'mixture': {'experience': 3, 'replay': 1},
                'gradient_accumulation': 4, 'rank': 4, 'alpha': 8}
    model = tiny_model(tokenizer, seed=2)
    trained, receipt = trainer.train(model, 'addition', [sequence], [sequence], settings)
    trainer.checkpoint(tmp_path / 'ckpt', trained, receipt, {'test': 'x'})
    fresh = tiny_model(tokenizer, seed=2)
    trainer.load_trainable(fresh, 'addition', settings, tmp_path / 'ckpt')
    ids = torch.tensor([sequence['input_ids'][:16]])
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, fresh(ids).logits, atol=1e-5)
    assert receipt['steps'] == 2 and receipt['trainable_parameters'] > 0


def test_execution_pins_sources_and_one_gpu_host_within_its_allowance():
    execution = read(ROOT / l4.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert l4.SCRIPT in execution['sources'] and execution['training_authorized']
    pinned = read(ROOT / 'config/experiments/assistant-growth-execution.json')
    keys = ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256', 'execution')
    assert all(execution[key] == pinned[key] for key in keys)
    probe = ('import os, sys; import neuroshard.evolution.assistant_general_l4 as m; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_documents_demonstration, neuroshard.evolution.assistant_general, '
             'neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_growth_run, '
             'neuroshard.evolution.assistant_rollout, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.granite_tokenizer, neuroshard.evolution.assistant_chat_data; m.spec(); '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']), sorted(set(imported) - set(execution['sources']))
    for path in l4.declared()['policies'].values():
        assert path in execution['sources']
    cloud = cloud_module()
    resources = cloud.resources(l4.PROFILE)
    assert resources['gpu'] and l4.PROFILE in cloud.GPU_PROFILES and 'upload' not in resources
    assert cloud.remote_command(l4.PROFILE)[1].endswith(l4.SCRIPT)
    assert json.loads(json.dumps(l4.declared()))['opens_confirmation'] is False
