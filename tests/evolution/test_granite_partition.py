import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from neuroshard.evolution.sharded import granite

BOUNDARIES = (0, 1, 3, 4)


def tiny_checkpoint(directory, dtype=torch.bfloat16, seed=0):
    from transformers import GraniteConfig, GraniteForCausalLM

    torch.manual_seed(seed)
    config = GraniteConfig(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                           embedding_multiplier=12.0, residual_multiplier=0.22, attention_multiplier=0.0625,
                           logits_scaling=10.0, pad_token_id=0, bos_token_id=1, eos_token_id=1,
                           rope_theta=10000000.0, initializer_range=0.1)
    GraniteForCausalLM(config).to(dtype).save_pretrained(directory, safe_serialization=True)
    return directory


def canonical(directory, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation='eager',
                                                local_files_only=True).eval()


def owners(directory, boundaries=BOUNDARIES, dtype=torch.bfloat16):
    read, _ = granite.checkpoint_reader(directory)
    config = granite.load_config(directory)
    return [granite.Partition(config, boundaries, rank, dtype=dtype).load(read)
            for rank in range(len(boundaries) - 1)]


@pytest.fixture
def checkpoint(tmp_path):
    return tiny_checkpoint(tmp_path / 'granite')


def prompt(length=11, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(2, 96, (1, length), generator=generator)


def test_partitioned_logits_equal_the_canonical_runtime_bit_for_bit(checkpoint):
    model, partitions = canonical(checkpoint), owners(checkpoint)
    ids = prompt()
    with torch.inference_mode():
        expected = model(ids).logits
        assert torch.equal(granite.forward(partitions, ids), expected)
        assert torch.equal(granite.forward(partitions, ids, last=True), model(ids, logits_to_keep=1).logits)


def test_warm_up_changes_nothing_the_owners_compute_afterwards(checkpoint):
    model, partitions = canonical(checkpoint), owners(checkpoint)
    for partition in partitions:
        partition.warm_up(lengths=(12, 1))
    ids = prompt()
    with torch.inference_mode():
        assert torch.equal(granite.forward(partitions, ids), model(ids).logits)


def test_cached_greedy_generation_matches_canonical_generate_token_for_token(checkpoint):
    model, partitions = canonical(checkpoint), owners(checkpoint)
    for seed in range(3):
        ids = prompt(seed=seed)
        with torch.inference_mode():
            expected = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                                      use_cache=True, max_new_tokens=16, eos_token_id=None, pad_token_id=0)
            tokens = granite.generate(granite.local_step(partitions), ids, 16, eos_ids=())
        assert tokens == expected[0, ids.shape[1]:].tolist()


def test_every_tensor_has_exactly_one_owner_and_no_owner_holds_the_backbone(checkpoint):
    _, names = granite.checkpoint_reader(checkpoint)
    partitions = owners(checkpoint)
    owned = [set(p.owned_names()) for p in partitions]
    assert set().union(*owned) == set(names) and sum(map(len, owned)) == len(names)
    total = sum(p.resident_bytes() for p in partitions)
    assert all(p.resident_bytes() < total for p in partitions)
    with pytest.raises(ValueError, match='another owner'):
        partitions[1].parameter('model.layers.0.mlp.up_proj.weight')
    with pytest.raises(ValueError, match='embedding'):
        partitions[2].embed(prompt())
    config = granite.load_config(checkpoint)
    for bad in ((0, 2, 2, 4), (0, 1, 3), (1, 4)):
        with pytest.raises(ValueError, match='partition the entire model'):
            granite.Partition(config, bad, 0)
    config._attn_implementation = 'sdpa'
    with pytest.raises(ValueError, match='eager'):
        granite.Partition(config, BOUNDARIES, 0)


def test_exported_shards_hold_only_owned_tensors_and_reload_identically(checkpoint, tmp_path):
    from safetensors.torch import load_file, save_file

    shards = tmp_path / 'shards'
    manifests = [granite.export(checkpoint, BOUNDARIES, rank, shards) for rank in range(3)]
    config = granite.load_config(checkpoint)
    loaded = [granite.load_partition(config, shards, rank)[0] for rank in range(3)]
    ids = prompt()
    with torch.inference_mode():
        assert torch.equal(granite.forward(loaded, ids), granite.forward(owners(checkpoint), ids))
    assert manifests[1]['tensors'] == loaded[1].owned_names()
    path = shards / 'partition-1.safetensors'
    values = load_file(path)
    values['model.layers.0.mlp.up_proj.weight'] = torch.zeros(128, 64, dtype=torch.bfloat16)
    save_file(values, str(path))
    manifest = json.loads((shards / 'partition-1.json').read_text())
    with pytest.raises(ValueError, match='differs from its manifest'):
        granite.load_partition(config, shards, 1)
    import hashlib
    manifest['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (shards / 'partition-1.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='exactly its owned tensors'):
        granite.load_partition(config, shards, 1)


def range_server(directory):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        requested = []

        def do_GET(self):
            data = (Path(directory) / self.path.lstrip('/')).read_bytes()
            begin, end = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
            Handler.requested.append((self.path, begin, end + 1))
            self.send_response(206)
            self.send_header('Content-Length', str(end + 1 - begin))
            self.end_headers()
            self.wfile.write(data[begin:end + 1])

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Handler.requested


def test_owners_fetch_only_their_verified_byte_ranges(tmp_path):
    import hashlib
    from transformers import GraniteForCausalLM

    source = tiny_checkpoint(tmp_path / 'one')
    split = tmp_path / 'split'
    GraniteForCausalLM.from_pretrained(source, dtype=torch.bfloat16).save_pretrained(split, max_shard_size='100KB')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(split).glob('*.safetensors')}
    assert len(files) > 1
    inventory = granite.tensor_inventory(split, files)
    with pytest.raises(ValueError, match='pinned digest'):
        granite.tensor_inventory(split, {**files, sorted(files)[0]: '0' * 64})
    server, requested = range_server(split)
    try:
        read = granite.http_ranges(f'http://127.0.0.1:{server.server_address[1]}')
        shards = tmp_path / 'fetched'
        manifests = [granite.fetch(inventory, BOUNDARIES, rank, shards, read) for rank in range(3)]
    finally:
        server.shutdown()
    config = granite.load_config(split)
    loaded = [granite.load_partition(config, shards, rank)[0] for rank in range(3)]
    with torch.inference_mode():
        assert torch.equal(granite.forward(loaded, prompt()), canonical(split)(prompt()).logits)
    data = sum(spec['end'] - spec['begin'] for spec in inventory['tensors'].values())
    assert sum(m['fetched_bytes'] for m in manifests) == data
    owned = {name: granite.owner(name, BOUNDARIES) for name in inventory['tensors']}
    ranges = {rank: [(spec['file'], spec['begin'], spec['end']) for name, spec in inventory['tensors'].items()
                     if owned[name] == rank] for rank in range(3)}
    assert all(any(f'/{file}' == path and b <= begin and end <= e for path, b, e in requested)
               for rank in range(3) for file, begin, end in ranges[rank])
    assert manifests[1]['fetched_bytes'] == sum(e - b for _, b, e in ranges[1])
    tampered = copy_inventory(inventory)
    name = next(n for n in tampered['tensors'] if owned[n] == 2)
    tampered['tensors'][name]['sha256'] = '0' * 64
    server, _ = range_server(split)
    try:
        with pytest.raises(ValueError, match='inventory digest'):
            granite.fetch(tampered, BOUNDARIES, 2, tmp_path / 'bad',
                          granite.http_ranges(f'http://127.0.0.1:{server.server_address[1]}'))
    finally:
        server.shutdown()


def copy_inventory(inventory):
    return json.loads(json.dumps(inventory))


def test_boundary_gradients_reproduce_the_complete_model(tmp_path):
    directory = tiny_checkpoint(tmp_path / 'fp32', dtype=torch.float32, seed=1)
    model = canonical(directory, dtype=torch.float32)
    partitions = owners(directory, dtype=torch.float32)
    ids = prompt(length=9)
    model(ids, labels=ids).loss.backward()
    for partition in partitions:
        partition.requires_grad_(True)
    # Each owner sees only a detached boundary tensor, as it would from a peer.
    inputs, outputs = [], []
    hidden = partitions[0].embed(ids)
    embedded = hidden
    for partition in partitions:
        boundary = hidden.detach().requires_grad_(True)
        inputs.append(boundary)
        hidden = partition(boundary)
        outputs.append(hidden)
    final = hidden.detach().requires_grad_(True)
    logits = partitions[0].logits(final)
    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
    loss.backward()
    gradient = final.grad
    for boundary, output in zip(reversed(inputs), reversed(outputs)):
        output.backward(gradient)
        gradient = boundary.grad
    embedded.backward(gradient)
    reference = dict(model.named_parameters())
    for partition in partitions:
        for name in partition.owned_names():
            expected = reference[name].grad
            assert torch.allclose(partition.parameter(name).grad, expected, rtol=1e-4, atol=1e-6), name
