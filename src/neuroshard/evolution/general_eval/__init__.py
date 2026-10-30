"""Protected general-assistant evaluation: reasoning, coding, writing, factual questions, planning, tool use.

Design and promotion rule: docs/GENERAL_CAPABILITY_EVAL.md. Modules:

- ``items``: item schema, loading, split digests and the salted sealed-split commitment;
- ``scorers``: one program scorer per area; ``sandbox``: isolated code execution for coding items;
- ``tools``: deterministic fixture tools for tool-use items (and the served documents tools);
- ``runner``: ``ModelSystem`` (any ``generate(messages, tools)``) and ``ServedSystem`` (the served Session
  with routing and units), the run loop, the JSON report and rescoring;
- ``served``: bindings for ``a4-general``; ``judge``: blind pairwise protocol and calibration;
- ``stats``: intervals and power; ``gate``: the promotion rule.

The development split (``data/development``) is open. No sealed split is, or may be, stored in this repository.
"""
