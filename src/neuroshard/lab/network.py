"""Five real CometBFT nodes exercising candidate membership and economics."""

import argparse
import base64
import copy
import datetime
import functools
import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from neuroshard.demo import client as wire, network as base, protocol, work
from neuroshard.lab import client, state, storage
from neuroshard.lab.app import execution_manifest


def wait_for(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError, KeyError) as exc:
            last = exc
        time.sleep(0.2)
    raise AssertionError(f"Candidate condition timed out: {last}")


def initialize(base_port, engine=None):
    home = Path(tempfile.mkdtemp(prefix="candidate-", dir=base.REPO / ".neuroshard"))
    config = base.initialize(home, base_port, engine)
    extra = home / "chain/node4"
    subprocess.run([config["engine"], "init", "--home", str(extra)], check=True, capture_output=True)
    key = subprocess.check_output([config["engine"], "show-node-id", "--home", str(extra)], text=True).strip()
    config["nodes"].append({"home": str(extra), "id": key, "p2p": base_port + 40,
                            "rpc": base_port + 41, "abci": base_port + 42})
    genesis = json.loads((home / "chain/node0/config/genesis.json").read_text())
    entries = []
    for i in range(4):
        owner = protocol.Identity.load_or_create(home / f"founder{i}.key")
        public, _ = client.consensus_identity(config["nodes"][i]["home"])
        entries.append({"owner": owner.public_key, "consensus_key": public,
                        "bond": 10 * state.PARAMS["bond_unit"], "liquid": 20_000_000})
        genesis["validators"][i]["power"] = "10"
    data = work.read_data(config["data"])
    genesis["app_state"] = {"manifest": execution_manifest(data), "validators": entries}
    genesis["consensus_params"]["evidence"].update(max_age_num_blocks=str(state.PARAMS["evidence_blocks"]),
                      max_age_duration=str(state.PARAMS["evidence_seconds"] * 1_000_000_000))
    template = (home / "chain/node0/config/config.toml").read_text()
    for i, node in enumerate(config["nodes"]):
        config_path = Path(node["home"]) / "config/config.toml"
        peers = ",".join(f'{p["id"]}@127.0.0.1:{p["p2p"]}' for j, p in enumerate(config["nodes"]) if i != j)
        text = template
        for section, option, value in [("", "proxy_app", f'"127.0.0.1:{node["abci"]}"'),
            ("p2p", "persistent_peers", json.dumps(peers)), ("p2p", "laddr", f'"tcp://127.0.0.1:{node["p2p"]}"'),
            ("rpc", "laddr", f'"tcp://127.0.0.1:{node["rpc"]}"')]:
            text = base.edit_config(text, section, option, value)
        config_path.write_text(text)
        (config_path.parent / "genesis.json").write_bytes(work.canonical(genesis))
        # Each existing node starts with its own corpus copy; the newcomer fetches it from peers below.
        node["data"] = str(Path(node["home"]) / "corpus.txt")
        if i < 4:
            Path(node["data"]).write_bytes(data)
    (home / "network.json").write_bytes(work.canonical(config))
    return config


def start_node(config, index):
    node = config["nodes"][index]
    base.launch(config, f"app{index}", [sys.executable, "-m", "neuroshard.lab.app", "--home", node["home"],
                "--data", node["data"], "--port", str(node["abci"])])
    base.launch(config, f"node{index}", [config["engine"], "start", "--home", node["home"]])


def ready(config):
    def condition():
        statuses = [wire.query(base.urls(config, i)[0]) for i in range(5)]
        sync = [wire.rpc(base.urls(config, i)[0], "status")["sync_info"]["catching_up"] for i in range(5)]
        if not any(sync) and all(s["height"] > 0 for s in statuses):
            if len({(s["round"], s["model_root"], work.digest(s["lease"])) for s in statuses}) == 1:
                for worker in base.urls(config)[1]:
                    wire.http(worker + "/identity")
                return statuses
    return wait_for(condition)


def public_validators(url, height):
    result = wire.rpc(url, "validators", {"height": str(height), "per_page": "100"})
    return {base64.b64decode(v["pub_key"]["value"]).hex(): int(v["voting_power"]) for v in result["validators"]}


def expect_rejection(url, tx, text):
    before = wire.query(url)
    try:
        wire.broadcast(url, tx)
    except wire.Rejected as exc:
        if text not in str(exc):
            raise AssertionError(f"Unexpected rejection: {exc}")
        after = wire.query(url)
        for key in ("model_root", "round", "issued"):
            assert before[key] == after[key]
        return str(exc)
    raise AssertionError("Adversarial transaction was accepted")


def retrieval(config):
    home = Path(config["home"])
    data = work.read_data(config["data"])
    spec = storage.publish(home / "peer-content", data)
    class Content(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass
    good = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Content, directory=str(home / "peer-content")))

    class Corrupt(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"forged checkpoint chunk")
        def log_message(self, *_):
            pass

    bad = ThreadingHTTPServer(("127.0.0.1", 0), Corrupt)
    for server in (good, bad):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        peers = [f"http://127.0.0.1:{server.server_port}" for server in (bad, good)]
        result = storage.fetch(spec, peers, config["nodes"][4]["data"])
        assert result["rejected_responses"] == len(spec["chunks"])
        assert Path(config["nodes"][4]["data"]).read_bytes() == data
        try:
            storage.fetch(spec, peers[:1], home / "must-not-exist.txt")
        except ValueError:
            assert not (home / "must-not-exist.txt").exists()
        else:
            raise AssertionError("Unavailable data was accepted")
        result["all_peers_corrupt_fails_closed"] = True
        return result
    finally:
        for server in (good, bad):
            server.shutdown()
            server.server_close()


def run(base_port=31650, output=None, evidence_binary=None):
    work.configure_cpu()
    config = initialize(base_port)
    home = Path(config["home"])
    url, workers = base.urls(config)
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Single-host, five real native consensus nodes, two pipeline workers, one inference provider",
              "chain_id": config["chain_id"], "parameters": state.PARAMS,
              "source_manifest": execution_manifest(work.read_data(config["data"]))}
    try:
        report["authenticated_retrieval"] = retrieval(config)
        for i in range(5):
            start_node(config, i)
        for i, port in enumerate(config["workers"]):
            base.launch(config, f"worker{i}", [sys.executable, "-m", "neuroshard.demo.worker", "--stage", str(i),
                "--port", str(port), "--key", str(home / f"worker{i}.key")])
        provider = protocol.Identity.load_or_create(home / "provider.key")
        provider_url = f"http://127.0.0.1:{base_port + 52}"
        base.launch(config, "provider", [sys.executable, "-m", "neuroshard.lab.provider", "--key", str(home / "provider.key"),
                                          "--port", str(base_port + 52)])
        initial = ready(config)[0]
        sponsor = protocol.Identity.load_or_create(home / "founder0.key")
        outsider = protocol.Identity.load_or_create(home / "worker0.key")
        assert outsider.public_key not in initial["accounts"]
        report["outsider_initial_balance"] = 0
        report["initial_validator_count"] = len(public_validators(url, initial["height"]))
        print("CANDIDATE: four validators and an unbonded fifth node are ready", flush=True)
        for _ in range(2):
            last_tx = client.mine(url, workers, sponsor)
        report["duplicate_rejection"] = expect_rejection(url, sponsor.sign(last_tx["body"]), "nonce")
        earned = wire.query(url)["accounts"][outsider.public_key]["balance"]
        assert earned == 2 * state.PARAMS["worker_budget"] // 2
        report["outsider_earned_training_atoms"] = earned
        receipt, new_key = client.bond(url, outsider, config["nodes"][4]["home"])
        requested = wire.query(url)["validators"][new_key]
        effective = requested["emit_at"] + 2
        report["join"] = {"bond_transaction_height": int(receipt["height"]), "effective_height": effective}
        assert new_key not in public_validators(url, int(receipt["height"]))
        wait_for(lambda: wire.query(url)["height"] >= effective + 2)
        powers = public_validators(url, effective)
        assert len(powers) == 5 and powers[new_key] == 1 and sum(powers.values()) == 41
        report["join"]["actual_voting_power"] = powers[new_key]
        print("CANDIDATE: a worker earned tokens and joined the actual validator set without an admission key", flush=True)

        # Prove the entrant can sign real native commits, and retain a valid offense height for the test below.
        address = state.consensus_address(new_key)
        def signed_height():
            height = wire.query(url)["height"]
            commit = wire.rpc(url, "commit", {"height": str(height)})["signed_header"]["commit"]
            return height if any(s["validator_address"] == address and s["block_id_flag"] == 2 for s in commit["signatures"]) else None
        wait_for(signed_height)
        client.mine(url, workers, sponsor)
        # Stop only its consensus engine: it must not receive the verifier share while absent.
        before_stop_height = wire.query(url)["height"]
        base.stop_process(config, "node4")
        wait_for(lambda: wire.query(url)["height"] >= before_stop_height + 2)
        before = wire.query(url)["accounts"][outsider.public_key]["balance"]
        client.mine(url, workers, sponsor)
        accepted_height = wire.query(url)["height"]
        wait_for(lambda: wire.query(url)["height"] >= accepted_height + 2)
        after = wire.query(url)["accounts"][outsider.public_key]["balance"]
        assert after - before == state.PARAMS["worker_budget"] // 2
        report["offline_validator_received_no_verifier_reward"] = True
        start_node(config, 4)
        ready(config)
        offense_height = wait_for(signed_height)

        withdrawal_request = client.transaction(url, outsider, "unbond", consensus_key=new_key)
        wire.broadcast(url, withdrawal_request)
        report["early_withdrawal_rejection"] = expect_rejection(url, client.transaction(url, outsider, "withdraw", consensus_key=new_key), "evidence window")
        removal = wire.query(url)["validators"][new_key]["emit_at"] + 2
        wait_for(lambda: wire.query(url)["height"] >= removal)
        assert new_key not in public_validators(url, removal)
        binary = Path(evidence_binary or base.REPO / ".neuroshard/tools/protocolprobe")
        generated = subprocess.run([str(binary), "--home", config["nodes"][4]["home"], "--rpc", url,
                                    "--height", str(offense_height)], check=True, capture_output=True, text=True)
        report["equivocation_evidence"] = json.loads(generated.stdout)
        wait_for(lambda: wire.query(url)["validators"][new_key]["amount"] < state.PARAMS["bond_unit"])
        validator = wire.query(url)["validators"][new_key]
        assert validator["amount"] == 187_500
        report["slash_after_removal_atoms"] = state.PARAMS["bond_unit"] - validator["amount"]
        print("CANDIDATE: genuine duplicate-vote evidence slashed the departed validator's still-locked bond", flush=True)
        wait_for(lambda: wire.query(url)["height"] >= validator["release_height"]
                 and wire.query(url)["time_ns"] > validator["release_time_ns"])
        before = wire.query(url)["accounts"][outsider.public_key]["balance"]
        wire.broadcast(url, client.transaction(url, outsider, "withdraw", consensus_key=new_key))
        withdrawn = wire.query(url)
        assert withdrawn["validators"][new_key]["status"] == "withdrawn"
        assert withdrawn["accounts"][outsider.public_key]["balance"] == before + 187_500 - state.PARAMS["fee"]
        report["withdrawn_after_both_evidence_windows"] = True

        public = [wire.http(worker + "/identity")["public_key"] for worker in workers]
        client.reserve(url, sponsor, public)
        job = wire.query(url)["lease"]
        malicious = client.training_claim(url, workers, sponsor)
        bad_body = copy.deepcopy(malicious["body"])
        bad_body["result_root"] = "0" * 64
        report["invalid_work_rejection"] = expect_rejection(url, sponsor.sign(bad_body), "full replay")
        wait_for(lambda: wire.query(url)["height"] > job["expires"])
        assert wire.query(url)["lease"] is None
        report["expired_reservation_burn_atoms"] = state.PARAMS["reservation_bond"]
        report["expired_work_rejection"] = expect_rejection(url, malicious, "not reserved")
        client.mine(url, workers, sponsor)

        request = {"prompt": "ROMEO:", "max_tokens": 12}
        client.reserve(url, sponsor, [provider.public_key], "infer", request, 100_000)
        task = wire.query(url, "/task")
        supplied = wire.http(provider_url + "/infer", {"task_id": task["lease"]["task_id"], "weights": task["weights"], "request": request})
        tx = client.transaction(url, sponsor, "submit", task_id=task["lease"]["task_id"],
                 result_root=work.digest(supplied["output"]), receipts=[supplied["receipt"]])
        wire.broadcast(url, tx)
        height = wire.query(url)["height"]
        wait_for(lambda: wire.query(url)["height"] >= height + 2)
        final = ready(config)[0]
        assert final["accounts"][provider.public_key]["balance"] == 80_000
        assert final["last_inference"] == supplied["output"]
        assert final["round"] == 5 and final["issued"] == 5_000_000
        report.update(final_round=final["round"], final_model_root=final["model_root"],
            initial_validation_loss=initial["validation_loss"], final_validation_loss=final["validation_loss"],
            initial_supply=final["initial_supply"], issued=final["issued"], burned=final["burned"],
            paid_inference_atoms=80_000, final_validator_count=len(public_validators(url, final["height"])))
        all_states = ready(config)
        assert len({s["model_root"] for s in all_states}) == 1
        assert len({work.digest(s["accounts"]) for s in all_states}) == 1
        assert len({work.digest(s["validators"]) for s in all_states}) == 1
        height = min(s["height"] for s in all_states)
        hashes = [wire.rpc(base.urls(config, i)[0], "block", {"height": str(height)})["block_id"]["hash"] for i in range(5)]
        assert len(set(hashes)) == 1
        report["shared_block"] = {"height": height, "hash": hashes[0]}
        report["all_five_nodes_agree"] = report["passed"] = True
        print("CANDIDATE: expiry, replay rejection, paid inference, supply accounting, and agreement passed", flush=True)
    finally:
        base.stop(config)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-port", type=int, default=31650)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-binary")
    args = parser.parse_args()
    result = run(args.base_port, args.output, args.evidence_binary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
