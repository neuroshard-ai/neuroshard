import json,os,subprocess,sys
from pathlib import Path

import pytest

from neuroshard.client import wire
from neuroshard.client.cli import atoms,neuro
from neuroshard.demo import protocol


def test_lightweight_entry_import_does_not_load_neural_runtime():
    result=subprocess.run([sys.executable,'-c','import neuroshard.client.cli,sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_wallet_signatures_match_native_protocol_and_key_creation_is_private(tmp_path):
    path=tmp_path/'account.key';wallet=wire.Wallet(path,create=True)
    body={'kind':'example','nonce':0,'prompt':'Unicode: café ☃ 😀','chain_id':'test'}
    envelope=wallet.sign(body);verified,owner=protocol.verify(envelope)
    assert verified==body and owner==wallet.public_key
    assert protocol.Identity.load_or_create(path).public_key==wallet.public_key
    assert protocol.transaction_id(envelope)==wire.transaction_id(envelope)
    if os.name=='posix':assert path.stat().st_mode&0o777==0o600
    assert wire.Wallet(path,create=True).public_key==wallet.public_key


@pytest.mark.parametrize(('text','value'), [('0.000001',1),('0.1',100000),('10',10000000),('100',100000000)])
def test_neuro_amounts_never_round_through_binary_float(text,value):
    assert atoms(text)==value and neuro(value)==text


@pytest.mark.parametrize('text',['nan','inf','0','-1','0.0000001','9007199254.740992'])
def test_invalid_budgets_are_rejected(text):
    with pytest.raises(ValueError):atoms(text)


def test_transport_and_json_reject_unsafe_inputs():
    for value in ['http://example.com','https://user:secret@example.com','file:///tmp/node','https://example.com/#fragment']:
        with pytest.raises(ValueError):wire.endpoint(value)
    for raw in [b'{"nonce":1,"nonce":2}',b'{"value":NaN}']:
        with pytest.raises(ValueError):wire.parse(raw)


def test_wallet_export_import_preserves_identity_without_printing_secret(tmp_path,capsys):
    from neuroshard.client.cli import main
    source=tmp_path/'source';target=tmp_path/'target';backup=tmp_path/'backup.json'
    main(['wallet','create','--home',str(source)])
    main(['wallet','export',str(backup),'--home',str(source)])
    main(['wallet','import',str(backup),'--home',str(target)])
    secret=(source/'account.key').read_text()
    assert secret==(target/'account.key').read_text()
    assert secret not in capsys.readouterr().out
    with pytest.raises(SystemExit):main(['wallet','import',str(backup),'--home',str(target)])
