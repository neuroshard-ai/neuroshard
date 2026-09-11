"""Restartable data windows, document grouping and train/evaluation exclusion."""
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from neuroshard.dataflow.store import canonical
from .objects import digest
from .schema import root, integer


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def fingerprint(text):
    words = re.findall(r'\w+', normalized(text))
    shingles = {' '.join(words[i:i+5]) for i in range(max(1,len(words)-4))}
    votes = [0]*64
    for shingle in shingles:
        value = int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8],'big')
        for bit in range(64):
            votes[bit] += 1 if value & (1<<bit) else -1
    return sum(1<<bit for bit,vote in enumerate(votes) if vote >= 0)


class Corpus:
    """A single registry groups duplicates before assigning any token window.

    SimHash distance <= 3 is a conservative near-duplicate heuristic, not a
    guarantee against semantic contamination. Source licenses remain explicit.
    """
    def __init__(self, home, store, tokenizer, sequence_length=64, target_mode='full', *, codec=None, max_windows=4):
        self.home, self.store, self.tokenizer = Path(home), store, tokenizer
        self.codec = codec
        self.tokenizer_root = codec.root if codec is not None else None
        self.max_windows = integer(max_windows,1,64)
        if codec is not None and (target_mode != 'response' or tokenizer is not codec.tokenizer):
            raise ValueError('A versioned corpus uses its codec and response-only targets')
        integer(sequence_length, 16, 256)
        self.sequence_length = sequence_length
        if target_mode not in ('full','response'):
            raise ValueError('Unknown training target mode')
        self.target_mode = target_mode
        self.home.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.home/'corpus.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, spec BLOB, cursor INTEGER);
            CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, signature TEXT, role TEXT, source TEXT, row INTEGER, object TEXT);
            CREATE TABLE IF NOT EXISTS bands (band TEXT, document TEXT, PRIMARY KEY(band,document));
            CREATE TABLE IF NOT EXISTS sequences (id TEXT PRIMARY KEY, document TEXT, role TEXT, object TEXT);
            CREATE TABLE IF NOT EXISTS evaluations (sequence TEXT PRIMARY KEY, candidate TEXT, reservation TEXT);
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value BLOB);
        ''')
        settings={'sequence_length':sequence_length,'target_mode':target_mode}
        if self.codec is not None:
            settings.update(tokenizer_root=self.tokenizer_root,max_windows=self.max_windows,
                            window_policy='all-assistant-turns-v1',evaluation_unit='document')
        old=self.db.execute('SELECT value FROM settings WHERE id=1').fetchone()
        if old and json.loads(old[0])!=settings:
            self.db.close()
            raise ValueError('Use a separate corpus home when changing the tokenizer or tokenization objective')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO settings VALUES (1,?)',(canonical(settings),))

    def register(self, source, initial_cursor=0):
        integer(initial_cursor,0,2**53-1)
        if (set(source) != {'repo','revision','split','license','role'} or
                re.fullmatch('[0-9a-f]{40}',source['revision']) is None or
                source['role'] not in ('train','retention','fresh','test','heldout') or not source['license']):
            raise ValueError('Pin source identity, license and data role explicitly')
        key = digest(canonical(source))
        self.store.put_json(source)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(key,canonical(source),initial_cursor))
        return key

    def collect(self, source_id, count=128, rows=None):
        integer(count,1,1024)
        row = self.db.execute('SELECT spec,cursor FROM sources WHERE id=?',(source_id,)).fetchone()
        if row is None:
            raise ValueError('Source is not registered')
        spec, cursor = json.loads(row[0]), row[1]
        if rows is None:
            from neuroshard.dataflow.collect import upstream_rows
            rows = upstream_rows(spec,self.home,cursor,count)
        accepted, scanned, rejected = [], 0, {'duplicate':0,'short':0,'invalid':0}
        coverage = {'response_tokens':0,'scored_tokens':0,'omitted_tokens':0,'truncated_documents':0}
        if self.codec is not None:
            rejected['incomplete_evaluation'] = 0
        with self.db:
            for item in rows:
                if scanned >= count:
                    break
                position = cursor+scanned
                scanned += 1
                messages = item.get('messages')
                if not isinstance(messages,list) or not messages or any(
                    not isinstance(m,dict) or m.get('role') not in ('user','assistant','system') or
                    not isinstance(m.get('content'),str) for m in messages
                ):
                    rejected['invalid'] += 1
                    continue
                text = '\n'.join(m['content'] for m in messages)
                try:
                    text_size=len(text.encode())
                except UnicodeError:
                    rejected['invalid'] += 1
                    continue
                if text_size > 256*1024:
                    rejected['invalid'] += 1
                    continue
                doc_id = digest(normalized(text).encode())
                role = ('retention','fresh','test')[int(doc_id,16)%3] if spec['role']=='heldout' else spec['role']
                signature = fingerprint(text)
                bands = [f'{i}:{(signature>>(16*i))&65535}' for i in range(4)]
                near = self.db.execute('SELECT DISTINCT d.signature FROM documents d JOIN bands b ON b.document=d.id WHERE b.band IN (?,?,?,?)',bands).fetchall()
                if self.db.execute('SELECT 1 FROM documents WHERE id=?',(doc_id,)).fetchone() or any((int(s,16)^signature).bit_count()<=3 for (s,) in near):
                    rejected['duplicate'] += 1
                    continue
                if self.codec is not None:
                    try:
                        prepared=self.codec.response_windows(messages,self.sequence_length//2,
                            self.sequence_length-self.sequence_length//2,self.max_windows)
                    except (ValueError,UnicodeError):
                        rejected['invalid'] += 1
                        continue
                    if role!='train' and prepared['truncated']:
                        rejected['incomplete_evaluation'] += 1
                        continue
                    windows=prepared['windows']
                    for field in ('response_tokens','scored_tokens','omitted_tokens'):
                        coverage[field] += prepared[field]
                    coverage['truncated_documents'] += int(prepared['truncated'])
                elif self.target_mode=='response':
                    from .batches import response_window
                    prepared=response_window(messages,self.tokenizer,self.sequence_length//2,self.sequence_length-self.sequence_length//2)
                    windows=[prepared] if prepared else []
                else:
                    tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                    maximum = 4 if role=='train' else 1
                    windows=[{'tokens':tokens[start:start+self.sequence_length]} for start in range(0,min(len(tokens)-self.sequence_length+1,maximum*self.sequence_length),self.sequence_length)]
                if not windows:
                    rejected['short'] += 1
                    continue
                document = {'id':doc_id,'source':source_id,'row':position,'messages':messages,'license':spec['license']}
                object_root = self.store.put_json(document)
                self.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?)',(doc_id,f'{signature:016x}',role,source_id,position,object_root))
                self.db.executemany('INSERT INTO bands VALUES (?,?)',[(b,doc_id) for b in bands])
                for prepared in windows:
                    identity = prepared['tokens'] if self.codec is None else {
                        'document':doc_id,'assistant_index':prepared['assistant_index'],
                        'target_start':prepared['target_start'],'target_end':prepared['target_end'],
                        'tokens':prepared['tokens'],'labels':prepared['labels'],'tokenizer_root':self.tokenizer_root}
                    seq_id = digest(canonical(identity))
                    window = self.store.put_json({'document':doc_id,**prepared,'role':role,'source':source_id})
                    inserted = self.db.execute('INSERT OR IGNORE INTO sequences VALUES (?,?,?,?)',(seq_id,doc_id,role,window)).rowcount
                    if inserted:
                        accepted.append(window)
            self.db.execute('UPDATE sources SET cursor=? WHERE id=?',(cursor+scanned,source_id))
        manifest = {'format':'neuroshard-data-window-v1','source':source_id,'start':cursor,'end':cursor+scanned,
                    'sequence_length':self.sequence_length,'target_mode':self.target_mode,'sequences':accepted,'rejected':rejected}
        if self.codec is not None:
            manifest.update(format='neuroshard-data-window-v2',tokenizer_root=self.tokenizer_root,
                            coverage=coverage,window_policy='all-assistant-turns-v1')
        return {'root':self.store.put_json(manifest),**manifest}

    def training(self, window_root, count, seed, replay_fraction=.25):
        integer(count,1,4096)
        if not 0 <= replay_fraction <= .75:
            raise ValueError('Invalid replay proportion')
        fresh = self.store.json(window_root)['sequences']
        if not fresh or any(self.store.json(k)['role']!='train' for k in fresh):
            raise ValueError('Training requires a nonempty training window')
        fresh_set = set(fresh)
        historical = [r[0] for r in self.db.execute("SELECT object FROM sequences WHERE role='train'") if r[0] not in fresh_set]
        def ranked(values, domain):
            return sorted(values,key=lambda k:digest(canonical([seed,domain,k])))
        fresh, historical = ranked(fresh,'new'),ranked(historical,'replay')
        result = []
        replay_count = int(count*replay_fraction) if historical else 0
        for i in range(count):
            use_replay = (i+1)*replay_count//count > i*replay_count//count
            pool = historical if use_replay else fresh
            result.append(pool[int(digest(canonical([seed,i])),16)%len(pool)])
        return result

    def reserve_evaluation(self, candidate, role, count, beacon):
        root(candidate)
        integer(count,32,1024)
        if role not in ('retention','fresh','test'):
            raise ValueError('Invalid evaluation role')
        reservations = self.db.execute('SELECT DISTINCT reservation FROM evaluations WHERE candidate=?',(candidate,)).fetchall()
        for (existing,) in reservations:
            saved = self.store.json(existing)
            if saved['role']==role and saved['beacon']==beacon:
                units=saved.get('documents',[]) if self.codec is not None else saved['sequences']
                if len(units)!=count:
                    raise ValueError('Evaluation reservation count changed')
                return existing
        # The chain must commit the candidate *before* supplying the selection
        # beacon. This local API alone does not make public examples secret.
        if self.codec is not None:
            return self._reserve_documents(candidate,role,count,beacon)
        available = [r[0] for r in self.db.execute('SELECT object FROM sequences s WHERE role=? AND NOT EXISTS (SELECT 1 FROM evaluations e WHERE e.sequence=s.object)',(role,))]
        selected = sorted(available,key=lambda k:digest(canonical([beacon,candidate,k])))[:count]
        if len(selected) != count:
            raise ValueError('Insufficient unused evaluation documents')
        manifest = {'candidate':candidate,'role':role,'beacon':beacon,'sequences':selected}
        reservation = self.store.put_json(manifest)
        with self.db:
            for key in selected:
                self.db.execute('INSERT INTO evaluations VALUES (?,?,?)',(key,candidate,reservation))
        return reservation

    def _reserve_documents(self,candidate,role,count,beacon):
        available = self.db.execute('''
            SELECT DISTINCT s.document FROM sequences s WHERE s.role=?
            AND NOT EXISTS (SELECT 1 FROM sequences other JOIN evaluations e
                ON e.sequence=other.object WHERE other.document=s.document)
        ''',(role,)).fetchall()
        selected = sorted((row[0] for row in available),key=lambda key:digest(canonical([beacon,candidate,key])))[:count]
        if len(selected)!=count:
            raise ValueError('Insufficient unused evaluation documents')
        documents=[]
        for document in selected:
            windows=[row[0] for row in self.db.execute('SELECT object FROM sequences WHERE document=? ORDER BY id',(document,))]
            documents.append({'document':document,'sequences':windows})
        sequences=[key for document in documents for key in document['sequences']]
        manifest={'format':'neuroshard-evaluation-documents-v1','candidate':candidate,'role':role,
                  'beacon':beacon,'tokenizer_root':self.tokenizer_root,'documents':documents,'sequences':sequences}
        reservation=self.store.put_json(manifest)
        with self.db:
            for key in sequences:
                self.db.execute('INSERT INTO evaluations VALUES (?,?,?)',(key,candidate,reservation))
        return reservation


class TextCorpus(Corpus):
    """The maintained corpus profile; legacy Corpus preserves old experiments."""
    def __init__(self,home,store,codec,sequence_length=128,max_windows=4):
        super().__init__(home,store,codec.tokenizer,sequence_length,'response',codec=codec,max_windows=max_windows)


def publish_window(corpus,window_root,destination):
    """Mirror an immutable licensed window, including its complete provenance.

    Destination can be the existing S3Store or another content-addressed store.
    S3 is a replica; dataset identity is the digest, independent of its provider.
    """
    store=corpus.store
    window=store.json(window_root)
    source_row=corpus.db.execute('SELECT spec FROM sources WHERE id=?',(window['source'],)).fetchone()
    if source_row is None:
        raise ValueError('Window source is not in the local provenance registry')
    source=json.loads(source_row[0])
    if source['role']!='train':
        raise ValueError('This publisher exports training windows only')
    source_root=store.put_json(source)
    keys={window_root,source_root}
    if window.get('tokenizer_root') is not None:
        codec_root=window['tokenizer_root']
        if codec_root!=corpus.tokenizer_root:
            raise ValueError('Window belongs to a different tokenizer contract')
        keys.update((codec_root,store.json(codec_root)['backend']))
    for key in window['sequences']:
        sequence=store.json(key)
        if sequence['source']!=source_root or sequence['role']!='train':
            raise ValueError('Window contains an unrelated or protected sequence')
        row=corpus.db.execute('SELECT object FROM documents WHERE id=?',(sequence['document'],)).fetchone()
        if row is None:
            raise ValueError('Missing document provenance')
        keys.update((key,row[0]))
    total=0
    for key in sorted(keys):
        raw=store.get(key)
        if destination.put(raw)!=key:
            raise ValueError('Replica changed the content identity')
        if destination.get(key)!=raw:
            raise ValueError('Replica read-back differs from the source')
        total+=len(raw)
    return {'window_root':window_root,'objects':len(keys),'bytes':total,'read_back_verified':True}
