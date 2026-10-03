#!/usr/bin/env python3
"""Check native templates/shapes with metadata and random weights, not a study."""

import argparse
import hashlib
import json
from pathlib import Path

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace
from neuroshard.evolution.granite_partition_plan import estimate
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, save, sha256


def main(metadata, output, partition):
    import torch
    from transformers import AutoTokenizer, GraniteConfig, GraniteForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(27092026)
    artifacts = read(ROOT / 'config/experiments/granite-reference-artifacts.json')
    metadata_digests = {}
    for name, spec in artifacts['models']['baseline']['files'].items():
        if name.endswith('.safetensors') or name == 'model.safetensors.index.json':
            continue
        raw = (metadata / name).read_bytes()
        digest = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if spec['algorithm'] != 'git-blob-sha1' or spec['bytes'] != len(raw) or spec['digest'] != digest:
            raise ValueError(f'metadata differs from pinned artifacts: {name}')
        metadata_digests[name] = hashlib.sha256(raw).hexdigest()
    config = read(metadata / 'config.json')
    tokenizer = AutoTokenizer.from_pretrained(metadata, local_files_only=True)
    policy_path = ROOT / 'config/experiments/assistant-workflow-policy.json'
    policy = read(policy_path)
    prompts = []
    episode_roots = []
    # Scripted goal-directed traces only exercise serialization and scoring.
    # They never become model outputs, training data or learning evidence.
    for case in data.cases('development'):
        texts = []
        def call(name, args):
            return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'
        for turn in case['turns']:
            goal = turn['expected']
            texts.extend([call('list_documents', {'project': goal['project']}),
                          '\n'.join(call('read_document', {'document_id': key}) for key in goal['source_ids']),
                          call('save_draft', goal), 'Draft saved locally.'])
        replies = iter(texts)
        def respond(messages, tools):
            prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
            ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            if len(ids) > policy['generation']['max_input_tokens']:
                raise ValueError('scripted trajectory exceeds input budget')
            text = next(replies)
            prompts.append({'sha256': identity(prompt), 'tokens': len(ids)})
            return {'text': text, 'terminated': True, 'executed': False,
                    'input_token_ids': ids, 'token_ids': [], 'prompt_sha256': identity(prompt)}
        result = workflow.execute(case, respond, policy)
        if not result['score']['passed']:
            raise ValueError('scripted environment outcome failed')
        episode_roots.append(identity(result['calls']))
    first = data.cases('development')[0]
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': first['turns'][0]['user']}]
    prompt = tokenizer.apply_chat_template(messages, tools=workspace.TOOLS, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
    tiny = GraniteConfig(vocab_size=config['vocab_size'], hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True, pad_token_id=100257, eos_token_id=100257,
                         **{key: config[key] for key in ('embedding_multiplier', 'attention_multiplier',
                            'residual_multiplier', 'logits_scaling', 'rope_theta')})
    model = GraniteForCausalLM(tiny).to(torch.bfloat16).eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
    if generated.shape[-1] != inputs.input_ids.shape[-1] + 1:
        raise ValueError('tiny native generation failed')
    config_sha = sha256(metadata / 'config.json')
    save(partition, {'format': 'neuroshard-granite-partition-plan/1', 'config': config,
                     'config_sha256': config_sha, 'estimate': estimate(config),
                     'execution_implemented': False, 'checklist_credit': False}, exclusive=True)
    save(output, {'format': 'neuroshard-assistant-workflow-preflight/1',
                 'policy_sha256': sha256(policy_path), 'model_config_sha256': config_sha,
                 'metadata_sha256': metadata_digests,
                 'runtime': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__},
                 'scripted_environment_cases': len(episode_roots), 'scripted_transcripts_root': identity(episode_roots),
                 'native_prompt_count': len(prompts), 'native_prompt_inventory_root': identity(prompts),
                 'maximum_scripted_prompt_tokens': max(p['tokens'] for p in prompts),
                 'tiny_random_bf16_native_generation_calls': 1, 'pretrained_model_generations': 0,
                 'training_performed': False, 'admission_evidence': False}, exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    args = parser.parse_args()
    main(args.metadata, args.output, args.partition)
