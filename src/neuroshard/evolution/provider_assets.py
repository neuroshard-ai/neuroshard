"""Restore only a provider's committed partitions from configured mirrors."""
from pathlib import Path
import time
from urllib.parse import urlsplit

from . import answering, expert_checkpoint, serving_graph
from .objects import Objects
from .reference_data import identity, sha256
from .schema import integer, root
from .sharded import retained_objects


def plan(graph, rank, policy=None):
    """Return local relative paths and commitments, never publisher-supplied paths."""
    serving_graph.validate(graph)
    integer(rank, 0, 2 + len(graph['experts']))
    files = {}

    def add(path, digest, size=None):
        path = Path(path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('An asset escaped the provider model directory')
        spec = {'sha256': root(digest), 'bytes': size}
        if size is not None:
            integer(size, 1, retained_objects.MAX_OBJECT_BYTES)
        if str(path) in files and files[str(path)] != spec:
            raise ValueError('Conflicting committed provider assets')
        files[str(path)] = spec

    if rank < 3:
        records = {name: spec for name, spec in expert_checkpoint.parent_records(graph['parent']).items()
                   if expert_checkpoint.owner(name, graph['parent']['boundaries']) == rank}
        for spec in graph['interpreter_assets']['partitions'][str(rank)]['tensors'].values():
            add('interpreter/'+spec['sha256']+'.safetensors', spec['sha256'], spec['bytes'])
    else:
        name = graph['descriptor']['rules'][rank-3]['id']
        records = graph['experts'][name]['tensors']
    for spec in records.values():
        add('objects/'+spec['sha256']+'.safetensors', spec['sha256'], spec['bytes'])
    for name, digest in graph['tokenizer']['files'].items():
        if Path(name).name != name:
            raise ValueError('A tokenizer file must have a plain filename')
        add('seed/'+name, digest)
    if 'answering' not in graph:
        return files
    key = graph['answering']['policy_root']
    add('objects/policies/'+key[:2]+'/'+key, key)
    if policy is None:
        return files
    adapter = policy.get('planner_weights')
    if rank == 2 and adapter is not None:
        add('objects/policies/'+adapter['sha256']+'.safetensors', adapter['sha256'], adapter['bytes'])
    semantic = policy.get('semantic_questions')
    if rank == 0 and semantic is not None:
        model = semantic['encoder']
        for name, digest in model['files'].items():
            if Path(name).name != name:
                raise ValueError('A semantic encoder file must have a plain filename')
            add('objects/policies/semantic-encoders/'+identity(model)+'/'+name, digest)
    reranker = policy.get('question_reranker')
    if reranker is not None and rank == reranker['owner']:
        model = reranker['model']
        for name, spec in model['files'].items():
            if Path(name).name != name:
                raise ValueError('A question reranker file must have a plain filename')
            add('objects/policies/question-rerankers/'+identity(model)+'/'+name, spec['sha256'], spec['bytes'])
    return files


def prepare(graph, profile, rank, home, source_home, inventory, mirrors, *,
            max_bytes, max_seconds=600, restore=retained_objects.restore):
    """Hash-check the graph, executor and each restored byte within local limits.

    The inventory supplies lengths for digest-only tokenizer/policy metadata.
    It cannot change a file's graph commitment or add an arbitrary destination.
    Mirror bases are local operator configuration; network requests cannot supply
    URLs, presigned credentials, Python modules or filesystem destinations.
    """
    from .sharded.graph_execution import FORMAT, PROFILE_FIELDS
    from .sharded.peer_wire import SOURCES
    serving_graph.fields(profile, PROFILE_FIELDS, 'Invalid provider executor profile')
    if profile['format'] != FORMAT or identity(profile) != graph['executor_root']:
        raise ValueError('The offered graph has a different executor')
    if any(name not in profile['sources'] for name in SOURCES):
        raise ValueError('The offered executor does not bind its provider transport')
    for name, digest in profile['sources'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(Path(source_home)/path) != root(digest):
            raise ValueError('The offered executor requires different installed source')
    integer(max_bytes, 1, 1024**4)
    integer(max_seconds, 1, 1800)
    if not isinstance(mirrors, list) or not 1 <= len(mirrors) <= 8:
        raise ValueError('Configure one to eight immutable object mirrors locally')
    for mirror in mirrors:
        parsed = urlsplit(mirror)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Configure mirror base URLs without credentials, queries or fragments')
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    deadline, restored = time.monotonic() + max_seconds, {}

    def install(files):
        sizes = {}
        for name, spec in files.items():
            size = spec['bytes'] if spec['bytes'] is not None else inventory.get(spec['sha256'])
            sizes[name] = integer(size, 1, retained_objects.MAX_OBJECT_BYTES)
        if sum(sizes.values()) > max_bytes:
            raise ValueError('The complete provider partition exceeds its local disk budget')
        for name, spec in files.items():
            if name in restored:
                continue
            target = home/name
            if not target.resolve().is_relative_to(home):
                raise ValueError('A local symlink escapes the provider model directory')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Provider asset preparation exceeded its time budget')
            result = restore(spec['sha256'], sizes[name],
                [mirror.rstrip('/')+'/'+spec['sha256'] for mirror in mirrors], target,
                attempts=4, max_seconds=remaining)
            restored[name] = {'sha256': spec['sha256'], 'bytes': sizes[name],
                              'readback_verified': result['readback_verified']}

    # Fetch a policy before the large tensors so missing auxiliary model sizes
    # or an oversized complete partition fail before those transfers begin.
    initial = plan(graph, rank)
    if 'answering' in graph:
        key = graph['answering']['policy_root']
        name = 'objects/policies/'+key[:2]+'/'+key
        install({name: initial[name]})
        policy = answering.load(graph, Objects(home/'objects/policies'))
        from .sharded.planned_graph import validate_configuration
        validate_configuration(answering.core(graph), policy, source_home)
        initial = plan(graph, rank, policy)
    install(initial)
    return {'graph': identity(graph), 'executor': identity(profile), 'rank': rank,
            'bytes': sum(spec['bytes'] for spec in restored.values()), 'files': restored}
