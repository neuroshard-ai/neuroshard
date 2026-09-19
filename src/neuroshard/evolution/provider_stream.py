"""Bounded, authenticated access to provisional customer-visible generation.

Only the latest text is retained in memory. Resumption replaces the draft, never
concatenates an unverifiable suffix. Native settlement remains authoritative.
"""
from collections import OrderedDict
import threading

from neuroshard.demo import protocol
from .reference_data import identity
from .schema import integer, root
from .serving_graph import fields

FORMAT = 'neuroshard-hosted-visible-stream-v1'


class StreamLog:
    def __init__(self, owner, snapshot, *, limit=16):
        self.owner, self.snapshot, self.limit = owner, snapshot, limit
        self.lock, self.rows = threading.Lock(), OrderedDict()

    def begin(self, job, chain_id, epoch):
        row = {'format': FORMAT, 'chain_id': chain_id, 'job_id': job['id'],
            'assignment_root': epoch, 'graph': identity(job['graph']),
            'request_root': identity(job['request']), 'tokenizer': job['graph']['tokenizer']['root'],
            'sequence': 1, 'status': 'preparing', 'text': '', 'verified': False}
        with self.lock:
            if (job['id'], epoch) in self.rows:
                raise ValueError('Cannot restart a visible stream in the same assignment epoch')
            self.rows[(job['id'], epoch)] = row
            while len(self.rows) > self.limit:
                self.rows.popitem(last=False)

    def emit(self, job_id, epoch, *, text=None, status='generating'):
        if status not in ('preparing', 'generating', 'generated', 'submitted', 'failed'):
            raise ValueError('Unknown provisional stream state')
        if text is not None and (not isinstance(text, str) or len(text.encode()) > 32768):
            raise ValueError('Bounded visible text required')
        with self.lock:
            row = self.rows.get((job_id, epoch))
            if row is None:
                return
            value = row['text'] if text is None else text
            if status == row['status'] and value == row['text']:
                return
            row.update(sequence=row['sequence'] + 1, status=status, text=value)

    def read(self, envelope):
        body, requester = protocol.verify(envelope)
        fields(body, {'format', 'chain_id', 'job_id', 'assignment_root', 'after'}, 'Invalid visible stream request')
        if body['format'] != FORMAT:
            raise ValueError('Unknown stream protocol')
        job_id, epoch = root(body['job_id']), root(body['assignment_root'])
        integer(body['after'], 0, 2**31)
        current = self.snapshot(job_id)
        job, lease = current['job'], current['lease']
        if (not job or not lease or current['chain_id'] != body['chain_id']
                or job['payer'] != requester or lease['assignment_root'] != epoch
                or lease['providers']['0']['owner'] != self.owner.public_key):
            raise ValueError('Stream access requires the customer and current coordinator assignment')
        with self.lock:
            row = self.rows.get((job_id, epoch))
            if row is not None and body['after'] > row['sequence']:
                raise ValueError('The requested stream cursor is ahead of this assignment')
            update = dict(row) if row is not None and row['sequence'] > body['after'] else None
        return self.owner.sign({'request': body, 'update': update})


from neuroshard.client.provider_wire import poll
