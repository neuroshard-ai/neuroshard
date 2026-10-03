from neuroshard.evolution import append_only_execution as execution
from neuroshard.evolution.append_only_growth import CONTRACT_IDENTITY


def test_execution_uses_fresh_pairs_and_the_frozen_rule():
    plan = execution.load_plan()
    data = execution.load_data(plan)
    pairs = [(row["a"], row["b"]) for rows in data["roles"].values() for row in rows]
    assert len(pairs) == len(set(pairs)) == 1280
    assert not set(pairs) & execution.excluded_pairs()
    assert plan["rule"] == CONTRACT_IDENTITY
    assert plan["train"] is True
    assert plan["gpu_launch_authorized"] is False
    assert plan["upgrade_public_0_4_0"] is False
    assert plan["gates"]["minimum_served_gain"] == 4
    assert plan["gates"]["minimum_protected"] == 1


def _rows(data, role, correct):
    rows = []
    for index, truth in enumerate(data["roles"][role]):
        passed = index < correct
        value = truth["answer"] if passed else "999"
        rows.append({"id": truth["id"], "answer": truth["answer"], "text": value,
                     "parsed_answer": value, "passed": passed, "terminated": True, "seconds": 1.0})
    return rows


def _arm(data, new, old):
    return {"development": _rows(data, "development", new), "retention": _rows(data, "retention", old),
            "peak_rss_bytes": 1000}


def test_served_gain_keeps_protected_answers_without_beating_the_control():
    plan = execution.load_plan()
    data = execution.load_data(plan)
    parent, expert, control = _arm(data, 0, 8), _arm(data, 10, 0), _arm(data, 11, 0)
    training = {"frozen_unchanged": True}
    spent = {"matched_budget": True, "optimization_cpu_seconds": 50}
    result = execution.score(plan, data, parent, expert, control, training, spent, 50)
    assert result["passed"]
    assert result["gates"]["protected_kept"]
    assert len(result["served_new"]["gained"]) == 10
    assert result["settlement_authorized"] is False
    weak = execution.score(plan, data, parent, _arm(data, 3, 0), control, training, spent, 50)
    assert not weak["passed"]
