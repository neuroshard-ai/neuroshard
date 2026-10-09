"""The accepted assistant on your own machine: a research preview, not a hosted service.

Granite 4.1 3B comes from its pinned Hugging Face revision; U1, L2, L3, A2's gate and the
router come from the project's GitHub release. Every file is checked against its pin
before use. Serving is the confirmation's: the router sends each user turn to drafting,
where A2's gate picks the parent or U1 with L3 once per conversation, or to scheduling,
U1 with L2. The tasks are the fictional workspaces the assistant was measured on. Nothing
leaves the machine except those downloads, and conversations are not training data.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
import time
from urllib.request import urlopen

RELEASE = 'config/assistant-preview.json'
SETS = ('drafting', 'scheduling', 'cross')
CHUNK = 1 << 20


def _root():
    from neuroshard.evolution.modular_reference_execution import ROOT

    return ROOT


def _read(name):
    from neuroshard.evolution.modular_reference_execution import read

    return read(_root() / name)


def pins():
    return _read(RELEASE)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            value.update(block)
    return value.hexdigest()


def matches(path, pin):
    return path.is_file() and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256']


def local_path(directory, name):
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).is_absolute() or '..' in parts or not 1 <= len(parts) <= 2:
        raise ValueError(f'unsafe release path: {name}')
    return Path(directory).joinpath(*parts)


def download(url, path, pin, opener=urlopen):
    """Stream one release asset beside its destination; keep it only if its size and digest match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', delete=False)
    partial = Path(handle.name)
    try:
        with handle, opener(url, timeout=60) as response:
            received = 0
            for block in iter(lambda: response.read(CHUNK), b''):
                received += len(block)
                if received > pin['bytes']:
                    raise ValueError(f'{path.name} is larger than its pin')
                handle.write(block)
        if not matches(partial, pin):
            raise ValueError(f'{path.name} differs from its pinned digest')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def fetch_modules(home, opener=urlopen, progress=print):
    release = pins()
    directory = Path(home) / 'units'
    for name, pin in release['files'].items():
        path = local_path(directory, name)
        if not matches(path, pin):
            progress(f'Downloading {name} ({pin["bytes"] / 2 ** 20:.1f} MiB)')
            download(release['base_url'] + pin['asset'], path, pin, opener)
    return directory


def fetch(home, progress=print):
    """The published modules, then the parent from its pinned revision."""
    from neuroshard.evolution import granite_reference as reference
    from neuroshard.evolution.modular_reference_execution import verify_artifacts

    fetch_modules(home, progress=progress)
    inventory = _read(reference.ARTIFACTS)['models']['baseline']
    size = sum(spec['bytes'] for spec in inventory['files'].values()) / 10 ** 9
    progress(f'Checking Granite 4.1 3B ({size:.1f} GB; downloaded from Hugging Face the first time)')
    verify_artifacts(Path(home) / 'baseline', inventory, download=True)


def runtime():
    """The pinned packages are required; a CPU without the published BF16 instructions is reported."""
    import importlib.metadata

    execution = _read(pins()['execution'])
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('the assistant preview runs on Linux x86_64')
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors'):
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found = None
        if found != execution['packages'][name]:
            raise ValueError(f'{name} {execution["packages"][name]} is required; '
                             'install docs/granite-reference-requirements.txt in this environment')
    flags = set(Path('/proc/cpuinfo').read_text().split())
    return {'cpu_matches_published': set(execution['required_cpu_flags']) <= flags}


class Assistant:
    """The parent, U1 with L3 for drafting, U1 with L2 for scheduling, A2's gate and the router."""

    def __init__(self, home, threads=None):
        execution = _read(pins()['execution'])
        threads = threads or min(execution['threads'], os.cpu_count() or 1)
        for key, value in execution['environment'].items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(threads) if key.endswith('_NUM_THREADS') else value
        import torch
        from neuroshard.evolution import assistant_growth_cohort3_eval as development
        from neuroshard.evolution import assistant_growth_confirm as confirmation
        from neuroshard.evolution import assistant_growth_run as growth
        from neuroshard.evolution import granite_reference as reference
        from neuroshard.evolution import granite_tokenizer
        from neuroshard.evolution.modular_reference_execution import verify_artifacts

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        home = Path(home)
        parent, units = home / 'baseline', home / 'units'
        for name, pin in pins()['files'].items():
            if not matches(local_path(units, name), pin):
                raise ValueError(f'{name} is missing or changed; run neuroshard assistant fetch')
        verify_artifacts(parent, _read(reference.ARTIFACTS)['models']['baseline'], download=False)
        gates = confirmation.verify(units, execution, development.needed('upgrade'))
        plan = _read(growth.PLAN)
        spec = growth.stage_spec(plan, _read(_read(plan['plan'])['cohort1']['learning']))
        self.units = development.SYSTEMS['upgrade']
        self.tokenizer, _ = granite_tokenizer.load(parent)
        self.parent, _ = reference.load_model(parent, 'baseline')
        self.models = development.load_units(lambda: reference.load_model(parent, 'baseline')[0], spec, units,
                                             execution, self.units)
        self.a2_gate, self.turn_gate = gates['a2']['arms']['update']['gate'], gates['router']['gate']
        self.policies = development.policies()
        self.prefix = None
        self.threads = threads

    def select(self, turn, user):
        """The router's route for one user turn, from the frozen parent's features."""
        from neuroshard.evolution import assistant_routing as routing
        from neuroshard.evolution import assistant_selector as selector

        drafting = self.policies['drafting']
        if self.turn_gate.get('feature') == 'message-mean':
            if self.prefix is None:
                self.prefix = routing.message_prefix(self.parent, self.tokenizer, drafting, 'cpu')
            feature = routing.message_feature(self.parent, self.tokenizer, drafting, user, 'cpu', self.prefix)
        else:
            feature = routing.turn_feature(self.parent, self.tokenizer, drafting, user, 'cpu')
        return 'scheduling' if selector.choose(self.turn_gate, feature) else 'drafting'

    def routes(self, case):
        """A new conversation's responders; A2's gate picks the drafting model once, from the first turn."""
        from neuroshard.evolution import assistant_experience_run as accelerator
        from neuroshard.evolution import assistant_selector as selector
        from neuroshard.evolution.assistant_serving import cached_responder

        drafting, scheduling = self.policies['drafting'], self.policies['scheduling']
        drafting_unit, scheduling_unit = self.units
        arm = selector.choose(self.a2_gate, accelerator.boundary_feature(self.parent, self.tokenizer, drafting, case, 'cpu'))
        return {'drafting': (cached_responder(self.models[drafting_unit] if arm else self.parent, self.tokenizer,
                                              drafting), drafting),
                'scheduling': (cached_responder(self.models[scheduling_unit], self.tokenizer, scheduling), scheduling)}

    def episodes(self, cases):
        """Complete conversations, served exactly as the confirmation served them."""
        from neuroshard.evolution import assistant_growth_eval as stage1

        return stage1.routed_episodes(self.parent, self.models, self.tokenizer, None, self.a2_gate, self.turn_gate,
                                      cases, units=self.units, route_policies=self.policies)


class Conversation:
    """One conversation in one workspace, served turn by turn as the confirmation served it.

    Each message re-executes the conversation from its recorded replies, which rebuilds the
    workspace exactly, and only the new turn reaches the model. A failed turn ends the
    conversation, as it ends an evaluated episode.
    """

    def __init__(self, world, routes, select):
        self.world, self.make_routes, self.route_of = world, routes, select
        self.users, self.recorded, self.chosen = [], [], []
        self.routes = self.result = None
        self.ended = False

    def say(self, text):
        from neuroshard.evolution import assistant_routing as routing
        from neuroshard.evolution.modular_reference_execution import identity

        if self.ended:
            raise ValueError('This conversation has ended; /new starts another.')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('A message must be non-empty text.')
        case = {'id': 'assistant-preview', 'world': self.world,
                'turns': [{'user': user} for user in [*self.users, text]]}
        if self.routes is None:
            self.routes = self.make_routes(case)
        recorded, earlier = iter(self.recorded), len(self.users)

        def replaying(name):
            respond, policy = self.routes[name]

            def call(messages, tools):
                previous = next(recorded, None)
                if previous is None:
                    return respond(messages, tools)
                if previous['request_sha256'] != identity({'messages': messages, 'tools': tools}):
                    raise ValueError('the conversation no longer replays from its recorded replies')
                return previous
            return call, policy

        result = routing.execute(case, {name: replaying(name) for name in self.routes},
                                 lambda turn, user: self.chosen[turn] if turn < earlier else self.route_of(turn, user),
                                 _rescore=False)
        if len(result['rounds']) != earlier + 1:
            raise ValueError('the conversation did not reach the new message')
        start = self.result['rounds'][-1]['call_count'] if self.result else 0
        self.users.append(text)
        self.recorded, self.chosen, self.result = result['generations'], [r['route'] for r in result['rounds']], result
        row = result['rounds'][-1]
        self.ended = bool(row['failure'])
        return {'route': row['route'], 'completed': row['completed'], 'final_text': row['final_text'],
                'failure': row['failure'], 'seconds': row['seconds'] + row['selection_seconds'],
                'calls': result['calls'][start:], 'drafts': row['snapshot']['drafts'],
                'meetings': row['snapshot'].get('meetings', {})}


def replay(assistant, name, limit=None, progress=print):
    """Published confirmation conversations re-run here, each compared with its published episode."""
    from neuroshard.evolution import assistant_growth_cohort3_confirm as confirmation

    release = pins()
    which = 'drafting' if name == 'drafting' else 'calendar'
    cases = confirmation.opened(_read(release['execution']), which)[name][:limit]
    published = {row['id']: row for row in _read(release['published'][which])['reply']['episodes'][name]}
    rows = []
    for case in cases:
        episode, before = assistant.episodes([case])[0], published[case['id']]
        row = {'set': name, 'id': case['id'], 'family': case.get('family'), 'passed': episode['score']['passed'],
               'published_passed': before['score']['passed'], 'routes': episode['score']['routes'],
               'tokens_identical': ([g['token_ids'] for g in episode['generations']]
                                    == [g['token_ids'] for g in before['generations']]),
               'seconds': episode['seconds']}
        rows.append(row)
        progress(row)
    return rows


def workspace(policies, sample=0, path=None):
    """A sample workspace from the published cross set, with its own requests as suggestions, or your own."""
    from neuroshard.evolution import assistant_growth_cohort3_confirm as confirmation
    from neuroshard.evolution import assistant_routing as routing

    if path is not None:
        world, suggestions = json.loads(Path(path).read_text()), []
    else:
        cases = confirmation.opened(_read(pins()['execution']), 'calendar')['cross']
        if not 0 <= sample < len(cases):
            raise ValueError(f'sample must be 0 to {len(cases) - 1}')
        world, suggestions = cases[sample]['world'], [turn['user'] for turn in cases[sample]['turns']]
    routing.workspace(list(policies.values())).Workspace(world)
    return world, suggestions


def show(world, suggestions, write=print):
    documents, calendars = world['documents'], world.get('calendars', [])
    write(f'Workspace: {len(documents)} documents, {len(calendars)} team calendars.')
    for document in sorted(documents, key=lambda d: (d['project'], d['revision'])):
        write(f"  {document['id']}  {document['title']}, revision {document['revision']}, {document['status']}")
    for calendar in calendars:
        days = sorted(calendar['busy'])
        write(f"  {calendar['team']} calendar, {days[0]} to {days[-1]}")
    for index, text in enumerate(suggestions):
        write(('Try: ' if index == 0 else 'Then: ') + text)
    write('Type a request. /new starts over in this workspace, /quit leaves. Nothing you type leaves this machine.')


def render(reply, write=print):
    write(f"assistant ({reply['route']}, {reply['seconds']:.0f} s): {reply['final_text'] or '(no reply)'}")
    if reply['calls']:
        write('  tools: ' + ', '.join(row['call']['name'] for row in reply['calls']))
    for kind in ('drafts', 'meetings'):
        for project, value in reply[kind].items():
            fields = '; '.join(f"{key} {', '.join(map(str, value[key])) if isinstance(value[key], list) else value[key]}"
                               for key in sorted(value) if key not in ('project', 'revision'))
            write(f'  {kind[:-1]} for {project}: {fields}')
    if reply['failure']:
        write(f"  stopped: {reply['failure']}. This conversation is over; /new starts another.")


def chat(assistant, sample=0, path=None, read_line=input, write=print):
    world, suggestions = workspace(assistant.policies, sample, path)
    show(world, suggestions, write)
    conversation = Conversation(world, assistant.routes, assistant.select)
    while True:
        try:
            text = read_line('you> ').strip()
        except EOFError:
            return
        if text in ('/quit', '/exit'):
            return
        if text == '/new':
            conversation = Conversation(world, assistant.routes, assistant.select)
            write('New conversation in the same workspace.')
        elif text:
            try:
                render(conversation.say(text), write)
            except ValueError as error:
                write(str(error))


def register(subcommands, home):
    parser = subcommands.add_parser('assistant', help='Run the grown assistant on this machine (research preview)')
    actions = parser.add_subparsers(dest='assistant_action', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--home', type=lambda p: Path(p).expanduser(), default=home,
                        help='Directory for the parent model and modules')
    actions.add_parser('fetch', parents=[common], help='Download Granite 4.1 3B and the published modules, checking every file')
    serving = argparse.ArgumentParser(add_help=False, parents=[common])
    serving.add_argument('--threads', type=int, help='CPU threads (default: up to 8)')
    again = actions.add_parser('replay', parents=[serving], help='Re-run published confirmation conversations and compare')
    again.add_argument('--set', choices=SETS, default='drafting')
    again.add_argument('--limit', type=int, default=3, help='Conversations to run; 0 runs the whole set')
    talk = actions.add_parser('chat', parents=[serving], help='Talk to the assistant in a sample or your own workspace')
    talk.add_argument('--sample', type=int, default=0, help='Sample workspace number from the published cross set')
    talk.add_argument('--world', type=Path, help='Your own workspace JSON: documents and optional team calendars')


def run(args):
    if args.assistant_action == 'fetch':
        fetch(args.home)
        print('Ready. Next: neuroshard assistant replay, or neuroshard assistant chat')
        return
    if (args.threads is not None and args.threads < 1) or (args.assistant_action == 'replay' and args.limit < 0):
        raise ValueError('threads must be positive and limit must not be negative')
    status = runtime()
    if not status['cpu_matches_published']:
        print('This CPU lacks the AMX BF16 instructions of the published runs: replies are slower and may differ.')
    print('Loading Granite 4.1 3B three times (parent, drafting, scheduling); peak memory is about 24 GB.')
    assistant = Assistant(args.home, args.threads)
    if args.assistant_action == 'chat':
        chat(assistant, args.sample, args.world)
        return
    rows = replay(assistant, args.set, args.limit or None, progress=lambda row: print(
        f"{row['id']}  {row['family']}  {'passed' if row['passed'] else 'failed'} "
        f"(published {'passed' if row['published_passed'] else 'failed'}), "
        f"{'same tokens' if row['tokens_identical'] else 'different tokens'}, {row['seconds']:.0f} s"))
    published = _read(pins()['published']['summary'])['report']['sets'][args.set]
    print(f"Here: {sum(r['passed'] for r in rows)}/{len(rows)} passed, {sum(r['tokens_identical'] for r in rows)}/{len(rows)}"
          f" token for token. Published, whole set: {published['upgrade']}/{published['cases']}.")
    saved = Path(args.home) / 'replays' / f'{args.set}-{time.strftime("%Y%m%dT%H%M%S")}.json'
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps({'runtime': status, 'threads': assistant.threads, 'rows': rows}, indent=2))
    print(f'Saved {saved}')
