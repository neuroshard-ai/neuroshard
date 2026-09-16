"""Reserved, bounded execution of the measured balanced learning recipe.

The numerical recipe is unchanged. Native windows add durable checkpoints and
closed boundary witnesses, allowing each auditor to replay one partition at a
time. Helpers also capture and replay generation and paired quality evaluation.
"""
import base64
import hashlib
import json
from pathlib import Path

from .. import balanced, consolidation, reference_data as data
from .. import portable_lifecycle as lifecycle
from . import guarded, portable, transcript
from .training import generate, score

FORMAT = 'neuroshard-native-balanced-execution-v1'
ROOT = Path(__file__).resolve().parents[4]
ADDED_SOURCES = ('src/neuroshard/evolution/sharded/native_execution.py',
                 'src/neuroshard/evolution/portable_lifecycle.py',
                 'src/neuroshard/evolution/portable_work.py', 'scripts/run_native_shards.py')


def descriptor(plan, prepared):
    return {'format': FORMAT, 'prepared': data.identity(prepared), 'job': balanced.job(prepared),
        'plan': data.identity(plan), 'reference_root': plan['parent']['checkpoint'],
        'sources': {name: data.sha256(ROOT / name) for name in (*balanced.SOURCES, *ADDED_SOURCES)}}


def check_descriptor(value, plan, prepared):
    if value != descriptor(plan, prepared):
        raise ValueError('Native executor differs from the committed numerical recipe or sources')
    balanced.committed_prepared(plan, prepared)


def quality_policy(plan, prepared, execution):
    """Replay the already selected result; this is not another independent test."""
    return {'format': FORMAT + '/quality-policy', 'executor_root': data.identity(execution),
        'prepared': data.identity(prepared), 'baseline': plan['baseline']['checkpoint'],
        'generation_tokens': plan['generation_tokens'], 'quality_gate': plan['quality_gate'],
        'roles': {role: prepared['roles'][role] for role in balanced.FINALS},
        'scope': 'Exact reproduction of the frozen balanced continuation and its exposed final evaluation'}


def window_binding(prepared, job, before, after, reference_root):
    return {'job': job, 'prepared': data.identity(prepared), 'input': data.identity(before),
        'output': data.identity(after), 'reference': reference_root,
        'start': before['step'], 'end': after['step'], 'boundaries': before['boundaries']}


def update_window(shard, teacher, optimizer, wire, rows, schedule, plan, start, end):
    origin = plan['parent']['step']
    if not origin <= start < end <= origin + plan['training']['steps'] or end - start > 4:
        raise ValueError('Execute one to four prescribed updates within the activated recipe')
    reports = []
    for step in range(start, end):
        index = step - origin
        assignment = schedule[index]
        reports.append(guarded.train_step(shard, teacher, optimizer, wire,
            [rows[i] for i in assignment['indices']], plan['training'], index, plan['microbatch'],
            kl_strength=plan['reference']['kl_strength'], margin_strength=plan['reference']['margin_strength'],
            margin_min=plan['reference']['margin_min'], margin_max=plan['reference']['margin_max']))
    return reports


def train_window(home, shard, teacher, optimizer, wire, rows, prepared, plan, before, end, births, job):
    home = Path(home)
    captured = transcript.Recorder(wire, home / 'transcript')
    reports = update_window(shard, teacher, optimizer, captured, rows, prepared['schedule'], plan, before['step'], end)
    after = portable.commit(home, shard, optimizer, wire, job, end, data.identity(before), births)
    captured.finish(window_binding(prepared, job, before, after, plan['parent']['checkpoint']))
    data.save(home / 'steps.json', reports)
    return after


def replay_window(home, shard, teacher, optimizer, rows, prepared, plan, before, after,
                  births, job, transcripts, witness):
    closed = transcript.validate(transcripts)
    rank = shard.rank
    binding = window_binding(prepared, job, before, after, plan['parent']['checkpoint'])
    if transcripts[rank]['binding'] != binding:
        raise ValueError('Native witness differs from the activated computation')
    replay = transcript.Replay(witness, transcripts[rank])
    update_window(shard, teacher, optimizer, replay, rows, prepared['schedule'], plan, before['step'], after['step'])
    replay.finish()
    manifest = portable.write(home, shard, optimizer, job, after['step'], births)
    if data.identity(manifest) != after['shards'][rank]:
        raise ValueError('Replayed shard manifest differs from the claimed output')
    return {'rank': rank, 'valid': True, 'binding': binding, 'transcript_root': closed}


def evaluate_side(shard, wire, plan, records, tokenizer):
    outcomes = {}
    for role in balanced.FINALS:
        rows = records[role]
        result = {'losses': score(shard, wire, rows), 'answers': []}
        wire.segment()
        if role in ('test-new', 'test-prior'):
            for row in rows:
                prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                tokens = generate(shard, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                result['answers'].append({'id': row['id'], 'output_ids': tokens, 'text': text,
                    'check': consolidation.check_answer(plan, row['task'], text, role)})
                wire.segment()
        outcomes[role] = result
    return outcomes


def quality_decision(plan, prepared, outcomes, tokenizer):
    decisions = {}
    for role in ('test-new', 'test-prior'):
        before, after = outcomes['baseline'][role], outcomes['candidate'][role]
        summary = balanced.generation(plan, prepared, role, before['answers'], after['answers'], tokenizer)
        passed = all(v['candidate_correct'] >= v['baseline_correct'] for v in summary['families'].values())
        if role == 'test-new':
            passed = (passed and summary['wins'] - summary['losses'] >= plan['quality_gate']['min_net_gain']
                and summary['one_sided_p'] < plan['quality_gate']['max_one_sided_p']
                and summary['families']['total']['candidate_correct'] >= plan['quality_gate']['min_correct_totals'])
        decisions[role] = {'passed': passed, 'generation': summary}
    retention = consolidation.retention(plan, prepared, 'retention',
        outcomes['baseline']['retention']['losses'], outcomes['candidate']['retention']['losses'])
    retention['passed'] = retention['upper'] <= plan['quality_gate']['retention_upper_at_most_nats']
    decisions['retention'] = retention
    return {'passed': all(value['passed'] for value in decisions.values()), 'decisions': decisions}


def generate_response(shard, wire, request):
    return generate(shard, wire, request['prompt_ids'], request['max_tokens'], request['eos_id'])


def service_report(claim, rank, transcripts):
    closed = validate_service_transcripts(transcripts)
    if closed != claim['record_root'] or transcripts[rank]['binding'] != lifecycle.transcript_binding(claim):
        raise ValueError('Service witness differs from its native statement')
    return {'rank': rank, 'valid': True, 'transcript_root': closed,
            'binding': {'statement': data.identity(lifecycle.service_statement(claim))}}


SEGMENTS = FORMAT + '/segmented-witness'


def validate_service_transcripts(rows):
    if rows and rows[0].get('format') == SEGMENTS:
        count = len(rows[0]['segments'])
        if not 2 <= len(rows) <= 512 or not 1 <= count <= 4096:
            raise ValueError('Invalid segmented service coverage')
        for rank, row in enumerate(rows):
            if (row['rank'] != rank or row['world'] != len(rows) or row['format'] != SEGMENTS
                    or row['binding'] != rows[0]['binding'] or len(row['segments']) != count):
                raise ValueError('Incomplete segmented service coverage')
        for index in range(count):
            group = [row['segments'][index] for row in rows]
            if group[0]['binding'] != {'segment': index}:
                raise ValueError('Service witness segment reordered')
            transcript.validate(group)
        return data.identity(rows)
    return transcript.validate(rows)


def exchange_manifests(wire, value):
    """Exchange bounded large manifests through the unchanged small-message wire.

    Quality evaluation can produce tens of megabytes of transcript metadata.
    Each transport message stays below the wire's 2 MiB cap; declared sizes,
    total allocation and content digests are checked before decoding JSON.
    """
    chunk_size, rank_limit, total_limit = 512 * 1024, 128 * 1024**2, 256 * 1024**2
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(raw) <= rank_limit:
        raise ValueError('Service manifest exceeds its per-partition bound')
    declarations = wire.exchange({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if not isinstance(declarations, list) or len(declarations) != wire.world:
        raise ValueError('Incomplete service manifest declarations')
    for row in declarations:
        if (not isinstance(row, dict) or set(row) != {'bytes', 'sha256'}
                or type(row['bytes']) is not int or not 0 < row['bytes'] <= rank_limit
                or not isinstance(row['sha256'], str) or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('Invalid service manifest declaration')
    if sum(row['bytes'] for row in declarations) > total_limit:
        raise ValueError('Service manifests exceed the aggregate allocation bound')
    buffers = [bytearray() for _ in declarations]
    for offset in range(0, max(row['bytes'] for row in declarations), chunk_size):
        pieces = wire.exchange(base64.b64encode(raw[offset:offset + chunk_size]).decode('ascii'))
        if not isinstance(pieces, list) or len(pieces) != wire.world:
            raise ValueError('Incomplete service manifest chunk coverage')
        for piece, row, output in zip(pieces, declarations, buffers):
            expected = min(chunk_size, max(0, row['bytes'] - offset))
            if not isinstance(piece, str) or len(piece) != 4 * ((expected + 2) // 3):
                raise ValueError('Service manifest chunk exceeds its declared size')
            try:
                decoded = base64.b64decode(piece, validate=True)
            except ValueError as error:
                raise ValueError('Invalid service manifest chunk encoding') from error
            if len(decoded) != expected:
                raise ValueError('Service manifest chunk differs from its declared size')
            output.extend(decoded)
    values = []
    for row, output in zip(declarations, buffers):
        if len(output) != row['bytes'] or hashlib.sha256(output).hexdigest() != row['sha256']:
            raise ValueError('Service manifest differs from its declared digest')
        values.append(json.loads(output))
    return values


class SegmentedRecorder:
    """Bound each quality witness to one generation or one role's loss scan."""
    def __init__(self, wire, home):
        self.wire, self.home = wire, Path(home)
        self.rank, self.world = wire.rank, wire.world
        self.current, self.segments = None, []

    def __getattr__(self, name):
        if name not in ('send', 'receive', 'exchange', 'sum'):
            raise AttributeError(name)
        if self.current is None:
            self.current = transcript.Recorder(self.wire, self.home / f'segment-{len(self.segments):06d}')
        return getattr(self.current, name)

    def segment(self):
        if self.current is None or len(self.segments) >= 4096:
            raise ValueError('Empty or excessive quality witness segment')
        self.segments.append(self.current.finish({'segment': len(self.segments)}))
        self.current = None

    def finish(self, binding):
        if self.current is not None:
            self.segment()
        result = {'format': SEGMENTS, 'rank': self.rank, 'world': self.world,
                  'binding': binding, 'segments': self.segments}
        data.save(self.home / 'transcript.json', result)
        return result


class SegmentedReplay:
    def __init__(self, home, row):
        self.home, self.row = Path(home), row
        self.rank, self.world = row['rank'], row['world']
        self.current, self.index = None, 0

    def __getattr__(self, name):
        if name not in ('send', 'receive', 'exchange', 'sum'):
            raise AttributeError(name)
        if self.current is None:
            if self.index >= len(self.row['segments']):
                raise ValueError('Replay requested an unrecorded operation')
            self.current = transcript.Replay(self.home / f'segment-{self.index:06d}', self.row['segments'][self.index])
        return getattr(self.current, name)

    def segment(self):
        if self.current is None:
            raise ValueError('Replay communication order differs')
        self.current.finish()
        self.current = None
        self.index += 1

    def finish(self):
        if self.current is not None:
            self.segment()
        if self.index != len(self.row['segments']):
            raise ValueError('Replay did not cover the complete communication window')
