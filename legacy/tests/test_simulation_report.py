from types import SimpleNamespace

from simulate_network import evaluate_report, summarize_nodes


def _args(**overrides):
    defaults = {
        "nodes": 2,
        "no_training": False,
        "min_online": None,
        "min_training_nodes": None,
        "min_total_steps": None,
        "min_max_peers": None,
        "min_total_neuro": None,
        "max_crashed": 0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_summarize_nodes_counts_training_and_peer_progress():
    summary = summarize_nodes(
        [
            {"status": "training", "steps": 10, "neuro": 0.1, "peers": 1},
            {"status": "ready", "steps": 0, "neuro": 0.2, "peers": 0},
            {"status": "pending", "steps": 0, "neuro": 0.0, "peers": 0},
        ],
        launched=2,
    )

    assert summary["online"] == 2
    assert summary["training"] == 1
    assert summary["total_steps"] == 10
    assert summary["total_neuro"] == 0.30000000000000004
    assert summary["max_peers"] == 1


def test_evaluate_report_passes_tiny_training_gate_defaults():
    report = {
        "summary": {
            "online": 2,
            "training": 2,
            "total_steps": 100,
            "total_neuro": 0.5,
            "max_peers": 1,
            "crashed": 0,
        }
    }

    passed, failures = evaluate_report(report, _args())

    assert passed is True
    assert failures == []
    assert report["checks"]["passed"] is True


def test_evaluate_report_fails_when_training_does_not_advance():
    report = {
        "summary": {
            "online": 2,
            "training": 0,
            "total_steps": 0,
            "total_neuro": 0.1,
            "max_peers": 1,
            "crashed": 0,
        }
    }

    passed, failures = evaluate_report(report, _args())

    assert passed is False
    assert any("training nodes" in failure for failure in failures)
    assert any("total steps" in failure for failure in failures)


def test_evaluate_report_allows_smoke_without_training():
    report = {
        "summary": {
            "online": 2,
            "training": 0,
            "total_steps": 0,
            "total_neuro": 0.1,
            "max_peers": 1,
            "crashed": 0,
        }
    }

    passed, failures = evaluate_report(report, _args(no_training=True))

    assert passed is True
    assert failures == []
