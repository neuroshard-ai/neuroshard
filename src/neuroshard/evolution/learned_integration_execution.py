"""Stage-1 CPU execution freeze for learned integration.

This document records the 135M seed hashes, eight general-retention identities,
and a single-process CPU host. After the freeze is committed, the CPU loop in
learned_integration_run.py may train and score development. It does not
authorize a GPU launch. It does not open confirmation.
"""
import json
import subprocess
from pathlib import Path

from neuroshard.evolution.learned_integration import (
    CONTRACT_IDENTITY, FORMAT, bind_method_freeze, bind_spec, load_spec,
)
from neuroshard.evolution.programming_expert import GENERAL_SHA
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES as SEED_FILES, MODEL_REPO, MODEL_REVISION

EXECUTION_FORMAT = FORMAT + '/execution'
HOST = 'cpu'
GENERAL_RETENTION = (
    {'id': 'fd5ee93a13cf6fb5410187d9478d133b24319751a27f0a0abeab680a622c0a7a',
     'row': 13712, 'group': 'conversation'},
    {'id': '4b72d5af3557e2997d547c3f7ed61e4956ee5eb1240c2fb79d629dcfa79f46f2',
     'row': 10926, 'group': 'conversation'},
    {'id': '116f4c2edfc1466c7367274797b6b354f7db84bec801e1213d582dc14e3f344c',
     'row': 17751, 'group': 'constraints'},
    {'id': '266fbc1a4554283d3affb63ec8ac3fe96217e0d43587e3c196d3e267578262ed',
     'row': 6645, 'group': 'constraints'},
    {'id': '9636a899292eaf94a681b1c1fbdfa930c1c4feead90c356202e3b4b61881e22c',
     'row': 4043, 'group': 'summary'},
    {'id': '4bb44901b10f99226bd59c529c980bc1e2e58956a1c2f2c9779cd6c38007211a',
     'row': 4233, 'group': 'summary'},
    {'id': '34f68c33f6168a08dd2177a3044505bb40a9c760c7c853132f0290d1642d2ad2',
     'row': 20506, 'group': 'rewrite'},
    {'id': '5dac4cf70c0938b0985a2ba098a50515c8c3a08920d4b54758eafecfe8ecb58d',
     'row': 20999, 'group': 'rewrite'},
)
EXECUTION_SOURCES = (
    'config/experiments/learned-integration.json',
    'config/experiments/learned-integration-method.json',
    'docs/LEARNED_INTEGRATION_EXECUTION.md',
    'scripts/run_learned_integration_stage1.py',
    'src/neuroshard/evolution/learned_integration_execution.py',
    'src/neuroshard/evolution/learned_integration_run.py',
    'src/neuroshard/evolution/seed.py',
)


def method_path():
    marker = Path('config/experiments/learned-integration-method.json')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / marker
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('learned-integration-method.json is not next to this source tree')


def load_method(path=None):
    return json.loads(Path(path or method_path()).read_text())


def general_retention():
    selected = []
    groups = []
    for item in GENERAL_RETENTION:
        if item['id'] != identity({'dataset': GENERAL_SHA, 'row': item['row']}):
            raise ValueError('General retention identity does not match its corpus row')
        selected.append(dict(item))
        groups.append(item['group'])
    if sorted(groups) != sorted(['conversation', 'conversation', 'constraints', 'constraints',
                                 'summary', 'summary', 'rewrite', 'rewrite']):
        raise ValueError('General retention must take two conversations from each group')
    return selected


def execution_freeze():
    method = load_method()
    return {
        'format': EXECUTION_FORMAT,
        'spec': CONTRACT_IDENTITY,
        'method': identity(method),
        'train': True,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'confirmation_opened': False,
        'confirmation_scored': False,
        'host': HOST,
        'distributed_runtime': False,
        'seed': {
            'repo': MODEL_REPO,
            'revision': MODEL_REVISION,
            'license': 'Apache-2.0',
            'files': dict(SEED_FILES),
            'note': 'Mechanism study. Not a 0.4.0 protocol upgrade.',
        },
        'general_retention': {
            'corpus_sha256': GENERAL_SHA,
            'documents': 8,
            'scoring': 'exact-parent-response-match',
            'identities_recorded': True,
            'scored': False,
            'rows': general_retention(),
        },
        'files': {name: sha256(name) for name in EXECUTION_SOURCES},
    }


def require_committed(paths):
    for path in paths:
        try:
            tracked = subprocess.check_output(['git', 'show', 'HEAD:' + path], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise ValueError('Execution source is not committed: ' + path) from exc
        if tracked != Path(path).read_bytes():
            raise ValueError('Execution source is not committed: ' + path)


def bind_execution(spec, method, execution, *, require_committed_sources=False):
    bind_spec(spec)
    bind_method_freeze(method, spec)
    expected = execution_freeze()
    if execution != expected:
        raise ValueError('Execution freeze does not match the committed stage-1 CPU freeze')
    if execution.get('gpu_launch_authorized') is not False:
        raise ValueError('Execution freeze does not authorize a GPU launch')
    if execution.get('train') is not True:
        raise ValueError('Stage-1 CPU freeze must authorize training on this host')
    if execution.get('host') != HOST:
        raise ValueError('Stage-1 host is CPU')
    if execution.get('confirmation_opened') is not False or execution.get('confirmation_scored') is not False:
        raise ValueError('Confirmation remains closed')
    if execution.get('general_retention', {}).get('scored') is not False:
        raise ValueError('General retention identities are recorded, not scored')
    if execution.get('seed', {}).get('files') != dict(SEED_FILES):
        raise ValueError('135M seed file hashes changed')
    if require_committed_sources:
        require_committed([*EXECUTION_SOURCES, 'config/experiments/learned-integration-execution.json'])
    return {
        'learned_integration': identity(spec),
        'method': identity(method),
        'execution': identity(execution),
        'host': HOST,
        'gpu_launch_authorized': False,
        'confirmation_opened': False,
    }


def authorize_cpu_run(spec, method, execution):
    binding = bind_execution(spec, method, execution, require_committed_sources=True)
    if execution.get('gpu_launch_authorized') is not False:
        raise ValueError('Execution freeze does not authorize a GPU launch')
    return binding
