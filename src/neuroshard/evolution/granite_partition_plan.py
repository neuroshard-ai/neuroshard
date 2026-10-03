"""Shape-derived storage/traffic estimates, not a Granite execution runtime."""


def estimate(config, sequence=6144, batch=1, bytes_per_element=2):
    if (config['model_type'] != 'granite' or not config['tie_word_embeddings']
            or config.get('attention_bias', False) or config.get('mlp_bias', False)
            or config['num_hidden_layers'] % 2 or min(sequence, batch, bytes_per_element) < 1):
        raise ValueError('unsupported planning profile')
    hidden, intermediate = config['hidden_size'], config['intermediate_size']
    layers, heads = config['num_hidden_layers'], config['num_attention_heads']
    kv_heads = config['num_key_value_heads']
    if hidden % heads:
        raise ValueError('invalid attention shape')
    head_dim = hidden // heads
    kv = kv_heads * head_dim
    embedding = config['vocab_size'] * hidden
    layer = 2 * hidden * hidden + 2 * hidden * kv + 3 * hidden * intermediate + 2 * hidden
    half = layers // 2
    owners = [embedding + half * layer, half * layer + hidden]
    return {
        'parameters': sum(owners), 'layer_parameters': layer,
        'owner_parameters': owners,
        'owner_weight_bytes': [n * bytes_per_element for n in owners],
        'owner_kv_bytes': [2 * half * batch * sequence * kv * bytes_per_element] * 2,
        'prefill_boundary_bytes': 2 * batch * sequence * hidden * bytes_per_element,
        'decode_boundary_bytes_per_token': 2 * batch * hidden * bytes_per_element,
        'last_eight_q_v_parameters': min(8, layers) * hidden * (hidden + kv),
        'last_eight_q_v_lora_rank16_parameters': min(8, layers) * 16 * (3 * hidden + kv),
        'placement': 'owner 0: tied embedding/head + first half; owner 1: second half + final norm',
        'path': [0, 1, 0],
        'required_native_semantics': {key: config[key] for key in (
            'embedding_multiplier', 'residual_multiplier', 'attention_multiplier',
            'logits_scaling', 'rope_theta')},
        'measured_peak_memory': False, 'measured_latency': False,
        'training_activations_optimizer_buffers_included': False,
        'network_framing_token_return_included': False,
    }
