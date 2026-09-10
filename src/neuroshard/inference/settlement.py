"""Recover uncertain native submission outcomes by their immutable transaction hash."""
import base64,hashlib,time
from neuroshard.demo import client as wire,work


def submit(rpc,envelope,timeout=180):
    raw=work.canonical(envelope);tx_hash=hashlib.sha256(raw).hexdigest().upper()
    try:
        result=wire.rpc(rpc,'broadcast_tx_sync',{'tx':base64.b64encode(raw).decode()},timeout=90)
        if result.get('code',0):raise wire.Rejected(result.get('log','Transaction rejected'))
    except OSError:
        # The node may have accepted the request before the connection was interrupted.
        # Never sign a fresh nonce or reserve a second task while its outcome is unknown.
        pass
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            result=wire.rpc(rpc,'tx',{'hash':base64.b64encode(bytes.fromhex(tx_hash)).decode(),'prove':False},timeout=15)
        except (OSError,wire.Rejected):time.sleep(.5);continue
        receipt=result.get('tx_result',{})
        if receipt.get('code',0):raise wire.Rejected(receipt.get('log','Committed transaction rejected'))
        return result
    raise TimeoutError(f'Native transaction {tx_hash} has no observed final result; inspect it before retrying')
