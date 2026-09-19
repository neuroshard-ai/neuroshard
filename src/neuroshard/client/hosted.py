"""Resumable native payment and visible streaming for the accepted shard graph.

Customer keys and the durable outbox remain local. Quotes and final results come
from a pinned local validating node; provisional drafts come from the assigned
coordinator through its ledger-pinned certificate. No model runtime is imported.
"""
import copy
import fcntl
from http.client import HTTPException
import json
from pathlib import Path
import secrets
import time

from . import wire, provider_wire
from .local_node import LocalNode
from neuroshard.evolution.reference_data import save
from neuroshard.evolution.schema import integer, root
from neuroshard.evolution.transactions import Outbox

FORMAT = 'neuroshard-hosted-customer-v1'


class Display:
    def __init__(self, structured):
        self.structured, self.draft = structured, ''

    def __call__(self, value):
        if self.structured:
            print(json.dumps(value, ensure_ascii=True), flush=True)
            return
        def clean(text):
            return ''.join(c for c in text if c in '\n\t' or c.isprintable())
        if 'quote' in value:
            quote = value['quote']
            print(f"Request: {value['request_id']}\nMaximum reserved: {quote['maximum_debit_atoms']/1_000_000:g} NEURO "
                  f"including {quote['verification_atoms']/1_000_000:g} for complete verification.")
            print(value['notice'], flush=True)
        elif value.get('status') == 'new_assignment':
            self.draft = ''
            print('\nDraft (unverified until native settlement):', flush=True)
        elif value.get('format') == provider_wire.FORMAT:
            text = clean(value['text'])
            if text.startswith(self.draft):
                print(text[len(self.draft):], end='', flush=True)
            elif text:
                print('\n[Revised draft]\n' + text, end='', flush=True)
            elif self.draft:
                print('\n[Draft withdrawn]', flush=True)
            self.draft = text
        elif 'native_result' in value:
            result = value['native_result']
            if result['status'] == 'completed':
                text = clean(result['text'])
                if text != self.draft:
                    print('\n' + text)
                print(f"\nSettled in block {result['height']}. Native request: {result['id']}", flush=True)
            else:
                print('\nNative result: ' + json.dumps(result, ensure_ascii=True), flush=True)
        elif 'notice' in value:
            print('\n' + value['notice'], flush=True)
        elif value.get('status') == 'pending':
            print('\nStill pending. Resume with ' + value['resume'], flush=True)
        elif 'status' in value:
            print('\n' + value['status'].replace('_', ' '), flush=True)


def conversation(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 31 or len(messages) % 2 != 1:
        raise ValueError('Use at most 16 user turns, alternating with settled assistant replies')
    for index, row in enumerate(messages):
        if (not isinstance(row, dict) or set(row) != {'role', 'content'}
                or row['role'] != ('assistant' if index % 2 else 'user')
                or not isinstance(row['content'], str) or not row['content'].strip()):
            raise ValueError('Require alternating nonempty conversation messages')
    if sum(len(row['content'].encode()) for row in messages) > 32768:
        raise ValueError('Conversation exceeds its byte bound; start a new session')


class Customer:
    def __init__(self, home, node, wallet, outbox):
        self.home, self.node, self.wallet, self.outbox = Path(home), node, wallet, outbox
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)

    def write(self, row):
        save(self.home/(root(row['id']) + '.json'), row)
        return row

    def load(self, key):
        row = wire.parse((self.home/(root(key) + '.json')).read_bytes())
        if (row['format'] != FORMAT or row['chain_id'] != self.node.chain_id
                or row['payer'] != self.wallet.public_key or row['id'] != key):
            raise ValueError('Saved request belongs to another chain or account')
        return row

    def prepare(self, messages, maximum, ceiling, *, session=None):
        conversation(messages)
        integer(maximum, 1, 256)
        integer(ceiling, 1, 2**60)
        quoted = self.node.query('/hosting/quote', {'question': messages, 'max_tokens': maximum})
        request_root = wire.digest({'messages': messages, 'max_tokens': maximum})
        if (quoted['format'] != 'neuroshard-hosted-inference-quote-v1'
                or quoted['chain_id'] != self.node.chain_id or quoted['request_root'] != request_root
                or quoted['max_tokens'] != maximum):
            raise ValueError('The quote changed the conversation, network or generation limit')
        parts = ('execution_atoms', 'provider_atoms', 'verification_atoms', 'transaction_fee_allowance_atoms')
        for name in parts:
            integer(quoted[name], 0, 2**60)
        if (quoted['maximum_debit_atoms'] != sum(quoted[name] for name in parts)
                or quoted['maximum_debit_atoms'] > ceiling):
            raise ValueError('Complete quote exceeds --max-price; nothing was submitted')
        if session is not None and any(session[k] != quoted[k] for k in ('graph', 'tokenizer')):
            raise ValueError('The accepted graph or tokenizer changed; start a new session explicitly')
        account = self.node.query('/account', {'public_key': self.wallet.public_key})
        if account['balance'] < quoted['maximum_debit_atoms']:
            raise ValueError('Balance cannot cover execution, providers, complete verification and fees')
        row = {'format': FORMAT, 'id': secrets.token_hex(32), 'chain_id': self.node.chain_id,
            'payer': self.wallet.public_key, 'quote': quoted, 'messages': copy.deepcopy(messages),
            'maximum_debit_atoms': ceiling, 'phase': 'quoted', 'budget_id': None, 'job_id': None}
        return self.write(row)

    def tick(self, row):
        """Advance one durable step. Unknown outcomes retain the original nonce."""
        if row['phase'] == 'finished':
            return {'status': 'finished', 'result': row['result']}
        quote, key = row['quote'], row['id']
        if row['phase'] in ('quoted', 'funding'):
            # A saved funding intent may already have committed. Recover that
            # exact envelope even if its earlier discovery quote has aged out.
            operation = key + ':fund'
            recorded = self.outbox.recorded(operation)
            if not recorded and self.node.query('/summary')['height'] > quote['valid_until']:
                row.update(phase='finished', result={'status': 'quote_expired', 'spent_atoms': 0})
                self.write(row)
                return {'status': 'finished', 'result': row['result']}
            row['phase'] = 'funding'
            self.write(row)
            self.outbox.send(operation, 'fund_audit', publisher=quote['publisher'], auditors=[],
                stage_limit=quote['stage_limit'], expires_in=quote['expires_in'])
            row.update(phase='awaiting_audit', budget_id=self.outbox.logical_id(operation))
            self.write(row)
        if row['phase'] == 'awaiting_audit':
            reservation = key + ':reserve'
            if self.outbox.recorded(reservation):
                # Completion can consume the audit budget before the customer
                # restarts. Recover the saved reservation before reading it.
                self.outbox.confirm(reservation)
                row.update(phase='serving', job_id=self.outbox.logical_id(reservation))
                self.write(row)
        if row['phase'] == 'awaiting_audit':
            audits = self.node.query('/auditing')
            budget = audits['budgets'].get(row['budget_id'])
            if budget is None:
                closed = next((x for x in audits['history'] if x['id'] == row['budget_id']), None)
                if closed is None:
                    raise ValueError('Audit outcome is outside retained history; inspect the saved transaction')
                row.update(phase='finished', result={'status': 'unreserved_audit_closed', 'audit': closed})
                self.write(row)
                return {'status': 'finished', 'result': row['result']}
            # Once signing began, recovery must confirm the original reservation
            # before cancellation: a dropped acknowledgement is not a failure.
            recorded = self.outbox.recorded(reservation)
            if not recorded and self.node.query('/summary')['height'] > quote['valid_until']:
                self.outbox.send(key + ':cancel-audit', 'cancel_audit', budget_id=row['budget_id'])
                return {'status': 'audit_cancelled'}
            weights = budget['voting_snapshot']['owners']
            accepted = sum(weights[owner] for owner, value in budget['auditors'].items() if value['bond'])
            if not recorded and 3*accepted <= 2*sum(weights.values()):
                return {'status': 'awaiting_complete_audit_funding'}
            self.outbox.send(reservation, 'lease_expert', graph=quote['graph'], question=row['messages'],
                max_tokens=quote['max_tokens'], offers=quote['offers'], max_price=quote['execution_atoms'],
                max_provider_fee=quote['provider_atoms'], audit_budget=row['budget_id'], expires_in=quote['expires_in'])
            row.update(phase='serving', job_id=self.outbox.logical_id(reservation))
            self.write(row)
        current = self.node.snapshot(row['job_id'], refresh=True)
        if current['result'] is not None:
            result = current['result']
            if result['id'] != row['job_id'] or result['graph'] != quote['graph']:
                raise ValueError('Native result changed the reserved graph or job')
            row.update(phase='finished', result=result)
            self.write(row)
            return {'status': 'finished', 'result': result}
        if current['job'] is None or current['lease'] is None:
            raise ValueError('Native job is outside retained history; inspect the saved reservation')
        if wire.digest(current['job']['request']) != quote['request_root']:
            raise ValueError('Native assignment changed the reserved conversation')
        return {'status': 'serving', 'snapshot': current}


def run(args, ceiling):
    config = wire.parse(args.hosted_config.read_bytes())
    node = LocalNode(config['node_rpc'], config['chain_id'], config['manifest_root'])
    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.home/'hosted-chat.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wallet = wire.Wallet(args.home/'account.key')
        box = Outbox(args.home/'hosted-chat.sqlite', node.url, node.chain_id, wallet)
        customer = Customer(args.home/'hosted-requests', node, wallet, box)
        connection, cursor, assignment = None, 0, None
        try:
            if args.resume:
                if args.prompt is not None or args.session is not None:
                    raise ValueError('Resume uses its saved prompt and session; do not supply replacements')
                row = customer.load(args.resume)
            else:
                if not args.prompt:
                    raise ValueError('Supply a prompt or --resume REQUEST_ID')
                if box.pending():
                    raise ValueError('Resume the existing pending request before signing another')
                session = wire.parse(args.session.read_bytes()) if args.session and args.session.exists() else None
                if session and (session['chain_id'] != node.chain_id or session['payer'] != wallet.public_key):
                    raise ValueError('Conversation belongs to a different network or account')
                messages = (session['messages'] if session else []) + [{'role': 'user', 'content': args.prompt}]
                if args.quote_only:
                    conversation(messages)
                    print(json.dumps(node.query('/hosting/quote', {'question': messages,
                        'max_tokens': args.max_tokens}), indent=2), flush=True)
                    return
                row = customer.prepare(messages, args.max_tokens, ceiling, session=session)
                row['session_path'] = str(args.session.resolve()) if args.session else None
                row['session_parent'] = session.get('last_request') if session else None
                customer.write(row)
            announce = Display(args.json)
            announce({'request_id': row['id'], 'quote': row['quote'],
                'notice': 'Prompts, conversation, neural-call tokens and final replies are public ledger data. '
                          'Providers and auditors process them. Drafts are unverified until native settlement.'})
            deadline = time.monotonic() + args.wait_seconds
            previous_status = None
            while time.monotonic() < deadline:
                update = customer.tick(row)
                if update['status'] != previous_status:
                    announce({'request_id': row['id'], 'status': update['status']})
                    previous_status = update['status']
                if update['status'] == 'finished':
                    result = update['result']
                    if result['status'] == 'completed' and row.get('session_path'):
                        path = Path(row['session_path'])
                        history = wire.parse(path.read_bytes()) if path.exists() else None
                        parent = history.get('last_request') if history else None
                        if not result['text'].strip():
                            announce({'notice': 'Empty response was settled; conversation history was not extended'})
                        elif parent not in (row.get('session_parent'), row['id']):
                            announce({'notice': 'This session already advanced; its later history was preserved'})
                        else:
                            save(row['session_path'], {'chain_id': node.chain_id, 'payer': wallet.public_key,
                                'graph': row['quote']['graph'], 'tokenizer': row['quote']['tokenizer'],
                                'last_request': row['id'], 'messages': row['messages'] +
                                    [{'role': 'assistant', 'content': result['text']}]})
                    announce({'request_id': row['id'], 'native_result': result})
                    return
                if update['status'] == 'serving':
                    lease = update['snapshot']['lease']
                    epoch = lease['assignment_root']
                    provider = lease['providers']['0']
                    if assignment != epoch:
                        if connection is not None:
                            connection.close()
                        assignment, cursor = epoch, 0
                        connection = provider_wire.PinnedConnection(provider['endpoint'], provider['certificate'],
                            timeout=2, allow_private=config.get('allow_private', False))
                        announce({'status': 'new_assignment', 'assignment_root': epoch, 'text': '', 'verified': False})
                    try:
                        draft = provider_wire.poll(connection, wallet, provider['owner'], node.chain_id,
                                                   row['job_id'], epoch, cursor)
                        if draft is not None:
                            if any(draft[k] != row['quote'][k] for k in ('graph', 'tokenizer', 'request_root')):
                                raise ValueError('Coordinator changed the visible conversation version')
                            cursor = draft['sequence']
                            announce(draft)
                    except (OSError, ValueError, HTTPException):
                        connection.close()
                        # A missing stream cannot establish a refund or a failed
                        # numerical claim. Continue reading the local ledger.
                time.sleep(.2)
            announce({'request_id': row['id'], 'status': 'pending', 'resume': '--resume ' + row['id']})
        finally:
            if connection is not None:
                connection.close()
            box.close()
