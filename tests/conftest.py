"""Repository-wide test options.

``NEUROSHARD_REQUIRE_ENGINE=1`` turns every skip for a missing CometBFT engine into a failure, so a CI job
that is meant to run the real-chain integration tests cannot pass by silently skipping them.
"""
import os

import pytest

# A missing engine. A test pinned to one past qualification's exact binary may still skip elsewhere, by name.
ENGINE_SKIP = ('cometbft is not installed', 'cometbft v0.38.26 is not installed', 'needs cometbft')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get('NEUROSHARD_REQUIRE_ENGINE') != '1' or not report.skipped:
        return
    reason = str(report.longrepr[-1] if isinstance(report.longrepr, tuple) else report.longrepr).lower()
    if any(word in reason for word in ENGINE_SKIP):
        report.outcome = 'failed'
        report.longrepr = f'required real-chain test skipped: {reason}'
