"""Local keys, public genesis declarations, and checksum-pinned node bootstrap."""

import base64
import datetime
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.request import urlopen

from neuroshard.demo import network, protocol, work
from neuroshard.lab import client, state
from neuroshard.lab.app import execution_manifest


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "docs/eval/data/input.txt"
if not DATA.is_file():
    DATA = Path(__file__).parent / "data/input.txt"


def engine_path(value=None):
    engine = network.engine_path(value)
    version = subprocess.check_output([engine, "version"], text=True).strip()
    if version != "0.38.26":
        raise ValueError("This release requires CometBFT 0.38.26")
    return engine


def create_keys(home, engine):
    home = Path(home)
    if not (home / "config/priv_validator_key.json").exists():
        if home.exists() and any(home.iterdir()):
            raise ValueError("Initialize in a new empty home; existing state is never replaced")
        home.mkdir(parents=True, exist_ok=True)
        subprocess.run([engine, "init", "--home", str(home)], check=True, capture_output=True)
    identity = protocol.Identity.load_or_create(home / "account.key")
    for name in ("config/priv_validator_key.json", "config/node_key.json", "data/priv_validator_state.json"):
        (home / name).chmod(0o600)
    return identity


def declaration(home, chain_id, bond, liquid, engine):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,49}", chain_id):
        raise ValueError("Use a chain ID of 3–50 lowercase letters, digits, or hyphens")
    identity = create_keys(home, engine)
    key, secret = client.consensus_identity(home)
    if state.integer(bond, state.PARAMS["bond_unit"]) % state.PARAMS["bond_unit"]:
        raise ValueError("Genesis bond must be a multiple of the voting unit")
    state.integer(liquid)
    body = {"domain": "neuroshard/genesis-declaration/v1", "chain_id": chain_id,
            "owner": identity.public_key, "consensus_key": key, "bond": bond, "liquid": liquid,
            "possession": secret.sign(state.possession_message(chain_id, identity.public_key, key, bond, 0)).hex()}
    return identity.sign(body)


def make_genesis(chain_id, declarations, output, profile="testnet", engine=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    work.configure_cpu()
    entries = []
    for envelope in declarations:
        body, owner = protocol.verify(envelope)
        if (set(body) != {"domain", "chain_id", "owner", "consensus_key", "bond", "liquid", "possession"}
                or body["domain"] != "neuroshard/genesis-declaration/v1" or body["chain_id"] != chain_id
                or body["owner"] != owner):
            raise ValueError("Invalid or foreign genesis declaration")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["consensus_key"])).verify(
            bytes.fromhex(body["possession"]), state.possession_message(chain_id, owner, body["consensus_key"], body["bond"], 0))
        entries.append({key: body[key] for key in ("owner", "consensus_key", "bond", "liquid")})
    if len(entries) < 4 or len({v["owner"] for v in entries}) != len(entries):
        raise ValueError("Genesis needs at least four distinct account declarations; ownership must also be disclosed")
    spec = execution_manifest(work.read_data(DATA), profile)
    state.genesis(chain_id, entries, spec)
    native = spec["native_consensus"]
    genesis = {"genesis_time": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "chain_id": chain_id, "initial_height": "1", "app_hash": "", "app_state": {"manifest": spec, "validators": entries},
        "validators": [{"address": state.consensus_address(v["consensus_key"]),
            "pub_key": {"type": "tendermint/PubKeyEd25519", "value": base64.b64encode(bytes.fromhex(v["consensus_key"])).decode()},
            "power": str(v["bond"] // spec["params"]["bond_unit"]), "name": ""} for v in entries],
        "consensus_params": {"block": {"max_bytes": str(native["block_max_bytes"]), "max_gas": "-1"},
            "evidence": {"max_age_num_blocks": str(native["evidence_blocks"]),
                "max_age_duration": str(native["evidence_duration_ns"]), "max_bytes": str(native["evidence_max_bytes"])},
            "validator": {"pub_key_types": ["ed25519"]}, "version": {"app": "0"}, "abci": {"vote_extensions_enable_height": "0"}}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "genesis.json").exists():
        raise ValueError("A genesis bundle already exists here")
    raw = work.canonical(genesis)
    (output / "genesis.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (output / "genesis.sha256").write_text(f"{digest}  genesis.json\n")
    (output / "declarations.json").write_bytes(work.canonical(declarations))
    (output / "corpus.txt").write_bytes(work.read_data(DATA))
    return {"chain_id": chain_id, "genesis_sha256": digest, "manifest_hash": work.digest(spec),
            "profile": profile, "genesis_validators": len(entries), "initial_supply": sum(v["bond"] + v["liquid"] for v in entries)}


def load_genesis(source, expected_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A published SHA-256 genesis digest is required")
    if source.startswith("https://"):
        with urlopen(source, timeout=20) as response:
            raw = response.read(work.MAX_MESSAGE_BYTES + 1)
    elif "://" in source:
        raise ValueError("Download genesis over HTTPS, or supply a local file")
    else:
        raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Genesis checksum mismatch")
    return protocol.parse_json(raw), raw


def peer(value, private=False):
    match = re.fullmatch(r"([0-9a-f]{40})@([a-zA-Z0-9.-]+):(\d{1,5})", value)
    if not match or not 1 <= int(match[3]) <= 65535:
        raise ValueError("Peer must be node-id@hostname:port")
    try:
        address = ipaddress.ip_address(match[2])
    except ValueError:
        if match[2].lower() == "localhost" and not private:
            raise ValueError("A public peer cannot use localhost")
    else:
        if not address.is_global and not private:
            raise ValueError("Use --private-network for local/private peer addresses")
    return value


def initialize(home, genesis_source, genesis_sha256, peers, engine=None, base_port=26656,
               advertise=None, private=False, api_host="127.0.0.1", trusted_height=0, trusted_hash=None):
    work.configure_cpu()
    home = Path(home).resolve()
    genesis, raw = load_genesis(genesis_source, genesis_sha256)
    profile = genesis["app_state"]["manifest"].get("profile", "lab")
    if genesis["app_state"]["manifest"] != execution_manifest(work.read_data(DATA), profile):
        raise ValueError("Local code, data, or numerical runtime differs from genesis")
    if not private and profile != "testnet":
        raise ValueError("Lab timing parameters cannot initialize a public node")
    if not 1024 <= base_port <= 65531:
        raise ValueError("Base port must leave room for four local services")
    created = datetime.datetime.fromisoformat(genesis["genesis_time"].replace("Z", "+00:00")).timestamp()
    age = time.time() - created
    if trusted_height or trusted_hash:
        if trusted_height <= 0 or not trusted_hash or not re.fullmatch(r"[0-9a-fA-F]{64}", trusted_hash):
            raise ValueError("A checkpoint requires both its positive height and block hash")
    elif not private and age > genesis["app_state"]["manifest"]["params"]["evidence_seconds"]:
        raise ValueError("Older stake histories require a recent independently trusted checkpoint")
    if any((home / path).exists() for path in ("node.json", "candidate.sqlite", "data/blockstore.db")):
        raise ValueError("Node is already initialized; run it without replacing its history")
    engine = engine_path(engine)
    identity = create_keys(home, engine)
    own_id = subprocess.check_output([engine, "show-node-id", "--home", str(home)], text=True).strip()
    peers = [peer(item, private) for item in peers]
    if advertise:
        peer(f"{own_id}@{advertise}:{base_port}", private)
    config_path = home / "config/config.toml"
    text = config_path.read_text()
    values = [("", "proxy_app", f'"127.0.0.1:{base_port + 2}"'), ("", "abci", '"grpc"'),
        ("", "log_level", '"info"'), ("rpc", "laddr", f'"tcp://127.0.0.1:{base_port + 1}"'),
        ("rpc", "unsafe", "false"), ("p2p", "laddr", f'"tcp://0.0.0.0:{base_port}"'),
        ("p2p", "external_address", json.dumps(f"{advertise}:{base_port}" if advertise else "")),
        ("p2p", "persistent_peers", json.dumps(",".join(peers))), ("p2p", "pex", "true"),
        ("p2p", "addr_book_strict", "false" if private else "true"),
        ("p2p", "allow_duplicate_ip", "true" if private else "false"),
        ("p2p", "max_num_inbound_peers", "40"), ("p2p", "max_num_outbound_peers", "10"),
        ("consensus", "timeout_commit", '"1s"'), ("consensus", "timeout_propose", '"3s"'),
        ("consensus", "timeout_prevote", '"1s"'), ("consensus", "timeout_precommit", '"1s"')]
    for section, name, value in values:
        text = network.edit_config(text, section, name, value)
    config_path.write_text(text)
    (home / "config/genesis.json").write_bytes(raw)
    (home / "corpus.txt").write_bytes(work.read_data(DATA))
    config = {"home": str(home), "engine": engine, "base_port": base_port, "api_host": api_host,
        "peers": peers, "advertise": advertise,
        "chain_id": genesis["chain_id"], "genesis_sha256": genesis_sha256, "profile": profile,
        "public_key": identity.public_key, "node_id": own_id,
        "trusted_checkpoint": {"height": trusted_height, "hash": trusted_hash.upper()} if trusted_hash else None}
    (home / "node.json").write_bytes(work.canonical(config))
    return config
