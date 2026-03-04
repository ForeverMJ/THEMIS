from run_experiment_integrated import _should_apply_repair_brief


def test_repair_brief_only_applies_to_blocking_targets():
    assert _should_apply_repair_brief({"target_symbol": "ForeignKey.check", "blocking": True}) is True
    assert _should_apply_repair_brief({"target_symbol": "ForeignKey.check", "blocking": False}) is False
    assert _should_apply_repair_brief({"target_symbol": "", "blocking": True}) is False
    assert _should_apply_repair_brief(None) is False
