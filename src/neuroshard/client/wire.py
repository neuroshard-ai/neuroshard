"""Canonical signing and bounded RPC transport shared with the native protocol."""
import base64,hashlib,json,os,secrets
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,urlopen

from neuroshard.core.crypto.ecdsa import derive_keypair_from_token,ecdsa_sign,ecdsa_verify

MAX_RESPONSE=8*1024*1024


class Rejected(ValueError):
    """A native query or transaction was explicitly refused."""


def verify(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {'body', 'public_key', 'signature'}:
        raise ValueError('Invalid signed envelope')
    public = envelope['public_key']
    if (not isinstance(public, str) or len(public) != 66 or not isinstance(envelope['body'], dict)
            or not isinstance(envelope['signature'], str) or len(envelope['signature']) > 160):
        raise ValueError('Invalid signed body, key or signature encoding')
    if not ecdsa_verify(canonical(envelope['body']).decode(), envelope['signature'], bytes.fromhex(public)):
        raise ValueError('Signature verification failed')
    return envelope['body'], public


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def parse(raw):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def invalid(_):raise ValueError('Nonfinite JSON number')
    if len(raw)>MAX_RESPONSE:raise ValueError('Response exceeds 8 MiB')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


class Wallet:
    def __init__(self,path,create=False):
        self.path=Path(path)
        if create:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            try:
                fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            except FileExistsError:pass
            else:
                with os.fdopen(fd,'w') as f:
                    f.write(secrets.token_hex(32));f.flush();os.fsync(f.fileno())
        if not self.path.is_file():raise ValueError('No local key yet. Run neuroshard wallet create.')
        token=self.path.read_text().strip()
        if len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('Key file is not a native 32-byte seed; legacy registration tokens cannot be migrated')
        self.key=derive_keypair_from_token(token)
        self.public_key=self.key.public_key_bytes.hex()

    def sign(self,body):
        return {'body':body,'public_key':self.public_key,
                'signature':ecdsa_sign(canonical(body).decode(),self.key.private_key_bytes)}


def endpoint(url):
    parsed=urlparse(url)
    if (parsed.username or parsed.password or parsed.fragment or parsed.query
        or not parsed.hostname or parsed.scheme not in ('https','http')):
        raise ValueError('Use an HTTPS endpoint without embedded credentials')
    if parsed.scheme=='http' and parsed.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Plain HTTP is supported only on loopback')
    return url.rstrip('/')


def http(url,body=None,timeout=30):
    endpoint(url)
    request=Request(url,data=canonical(body) if body is not None else None,headers={'Content-Type':'application/json','User-Agent':'NeuroShard/0.4'})
    try:
        with urlopen(request,timeout=timeout) as response:
            # The response location must preserve the same transport requirements.
            endpoint(response.geturl());return parse(response.read(MAX_RESPONSE+1))
    except __import__('urllib.error',fromlist=['HTTPError']).HTTPError as exc:
        try:detail=parse(exc.read(8192)).get('error',f'HTTP {exc.code}')
        except (ValueError,AttributeError):detail=f'HTTP {exc.code}'
        raise ValueError(str(detail)) from exc


def rpc(url,method,params=None,timeout=30):
    response=http(url,{'jsonrpc':'2.0','id':1,'method':method,'params':params or {}},timeout)
    if 'error' in response:raise Rejected(str(response['error']))
    return response['result']


def query(url,path='/summary',data=None):
    params={'path':path,'prove':False}
    if data is not None:params['data']=canonical(data).hex()
    result=rpc(url,'abci_query',params)['response']
    if result.get('code',0):raise ValueError(result.get('log','Query rejected'))
    return parse(base64.b64decode(result['value'],validate=True))


def broadcast(url,envelope):
    result=rpc(url,'broadcast_tx_sync',{'tx':base64.b64encode(canonical(envelope)).decode()},timeout=90)
    if result.get('code',0):raise ValueError(result.get('log','Transaction rejected'))
    return result


def transaction_id(envelope):return digest({k:envelope[k] for k in ('body','public_key')})
