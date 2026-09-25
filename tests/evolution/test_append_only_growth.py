from neuroshard.evolution.append_only_growth import CONTRACT_IDENTITY, load_spec, score_served, serve
from neuroshard.evolution.reference_data import identity


def row(name, passed):
    return {"id": name, "passed": passed}


def test_contract_is_frozen_and_does_not_train():
    spec = load_spec()
    assert identity(spec) == CONTRACT_IDENTITY
    assert spec["train"] is False
    assert spec["gpu_launch_authorized"] is False
    assert spec["later_execution"]["authorized"] is False
    assert spec["parent_result"]["expert_new"] == 10
    assert spec["parent_result"]["control_new"] == 11
    assert spec["reuses_opened_measurement"] is False


def test_protected_parent_answer_is_kept_and_a_miss_can_be_filled():
    parent = row("old", True)
    added = row("old", False)
    served, source = serve(parent, added, {"old"})
    assert served is parent and source == "parent"
    parent_miss = row("new", False)
    added_hit = row("new", True)
    served, source = serve(parent_miss, added_hit, {"old"})
    assert served is added_hit and source == "added"
    both_wrong = row("other", False)
    served, source = serve(both_wrong, row("other", False), set())
    assert served is both_wrong and source == "parent"


def test_growth_rule_rejects_a_lost_protected_answer():
    parent = [row("old", True), row("fresh", False)]
    added = [row("old", False), row("fresh", True)]
    kept = score_served(parent, added, ["old"])
    assert kept["passed_growth_rule"]
    assert kept["protected_lost"] == []
    assert kept["gained"] == ["fresh"]
    assert kept["served"][0]["source"] == "parent"
    assert kept["served"][1]["source"] == "added"
