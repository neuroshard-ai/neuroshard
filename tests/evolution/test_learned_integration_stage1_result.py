import json
from pathlib import Path

from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def test_stage1_development_failed_and_closed_confirmation():
    result = json.loads((ROOT / 'config/experiments/learned-integration-stage1-result.json').read_text())
    record = json.loads((ROOT / 'config/experiments/learned-integration-stage1-record.json').read_text())
    assert identity(result) == '07d4ac47aa7a5ef29bb5b240773478f247309976c2177bacf41ee89e683d4f83'
    assert identity(record) == 'bf17899c1cc1adc35e01541b248e8e3452868d4af042272f5b360656dc067b82'
    assert result['development']['parent_code'] == 0
    assert result['development']['expansion_code'] == 0
    assert result['development']['control_code'] == 5
    assert result['development']['general_preserved'] == 8
    assert result['development_gate']['passed'] is False
    assert result['development_gate']['next'] == 'stop'
    assert result['confirmation_opened'] is False
    assert result['confirmation_scored'] is False
    assert result['admission_evidence'] is False
    assert result['gpu_launch_authorized'] is False
    assert record['control_code_task_ids'] == [517, 733, 807, 896, 924]
    assert record['expansion_code_task_ids'] == []
    assert record['parent_code_task_ids'] == []
    assert all(row['exact_match'] for row in record['general'])
    assert len(record['general']) == 8
