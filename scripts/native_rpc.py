"""Transaction confirmation for isolated experiments with expensive CheckTx.

Submit one signed envelope, then look up its exact hash. Losing the submission
response does not prove rejection and must not trigger a newly signed payment.
"""
import base64
import hashlib
import time
from http.client import HTTPException

from neuroshard.dataflow.store import canonical
from neuroshard.demo import client as wire


def broadcast_finalized(url,envelope,timeout=180,*,rpc=wire.rpc,clock=time.monotonic,sleep=time.sleep):
    raw=canonical(envelope)
    tx_hash=hashlib.sha256(raw).hexdigest().upper()
    deadline=clock()+timeout
    try:
        result=rpc(url,'broadcast_tx_sync',{'tx':base64.b64encode(raw).decode()},timeout=min(timeout,30))
    except (OSError,HTTPException):
        # The signed transaction may already be executing. Keep its hash and
        # nonce; determine final acceptance from the committed transaction.
        pass
    else:
        if result.get('code',0):
            raise wire.Rejected(result.get('log','Transaction rejected in CheckTx'))
    while clock()<deadline:
        try:
            # JSON-RPC []byte arguments use base64; URL-query hex examples
            # cannot be copied into a JSON body. Returned Hash is hexadecimal.
            result=rpc(url,'tx',{'hash':base64.b64encode(bytes.fromhex(tx_hash)).decode(),'prove':False},
                       timeout=min(10,deadline-clock()))
        except wire.Rejected as exc:
            if 'not found' not in str(exc).lower():
                raise
        except (OSError,HTTPException):
            pass
        else:
            if str(result.get('hash','')).upper()!=tx_hash or int(result.get('height',0))<1:
                raise wire.Rejected('Transaction lookup returned an unrelated or uncommitted result')
            receipt=result['tx_result']
            if receipt.get('code',0):
                raise wire.Rejected(receipt.get('log','Transaction rejected at finalization'))
            return result
        sleep(min(.2,max(0,deadline-clock())))
    raise TimeoutError(f'Transaction outcome unknown after confirmation deadline; query hash {tx_hash}')
