# Voting Guide

This guide explains how to participate in NeuroShard governance by voting on proposals.

## Voting Power

Your voting power is determined by your **staked NEURO**:

```
1 NEURO staked = 1 vote
```

## Voter Rewards

Voters earn rewards for participating in governance:

```
Reward = 0.01% × your_stake
```

| Your Stake | Reward per Vote |
|------------|-----------------|
| 100 NEURO | 0.01 NEURO |
| 1,000 NEURO | 0.1 NEURO |
| 10,000 NEURO | 1 NEURO |

::: tip Why Stake-Weighted Rewards?
This incentivizes participation while ensuring larger stakeholders (who have more to lose) are adequately rewarded for taking time to review proposals.
:::

## Vote Options

| Option | Meaning |
|--------|---------|
| **YES** | Approve the proposal |
| **NO** | Reject the proposal |
| **ABSTAIN** | Participate without opinion (counts for quorum) |

## Thresholds

For a proposal to pass:

| Threshold | Requirement |
|-----------|-------------|
| **Approval** | 66% of voting stake must vote YES |
| **Quorum** | 20% of total network stake must vote |

### Example

```
Network total stake: 1,000,000 NEURO

Votes:
- YES:     150,000 NEURO (50%)
- NO:       50,000 NEURO (17%)
- ABSTAIN: 100,000 NEURO (33%)

Total voted: 300,000 NEURO (30% of network)

Result:
✓ Quorum reached (30% > 20%)
✗ Approval failed (50% < 66%)
→ REJECTED
```

## How to Vote

### Via CLI (Recommended)

```bash
# List proposals currently in voting
neuroshard-governance list --status voting

# View proposal details
neuroshard-governance show NEP-001 --full

# Cast your vote
neuroshard-governance vote NEP-001 yes \
  --reason "Improves training efficiency"

# Check voting results
neuroshard-governance results NEP-001
```

### Via Python SDK

```python
from neuroshard.core.governance import VotingSystem, VoteChoice

voting = VotingSystem(ledger=node.ledger)

# Cast vote
success, message = voting.cast_vote(
    nep_id="NEP-001",
    voter_node_id=node.node_id,
    choice=VoteChoice.YES,
    signature=node.sign(f"VOTE:NEP-001:YES"),
    reason="This improves training efficiency without harming economics",
)

print(message)
# "Vote recorded: yes with 500.00 stake"
```

### Via Website

1. Go to [neuroshard.com/governance](https://neuroshard.com/governance)
2. Log in with your account
3. Find the proposal in "Active Votes"
4. Use the CLI commands shown to cast your vote

### Via CLI

```bash
# View active proposals
neuroshard governance list --status voting

# Vote on a proposal
neuroshard governance vote NEP-001 --choice yes --reason "Improves efficiency"

# Check vote status
neuroshard governance status NEP-001
```

## Viewing Results

### Current Tally

```python
from neuroshard.core.governance import VotingSystem

voting = VotingSystem(ledger=node.ledger)
result = voting.get_vote_tally("NEP-001")

print(f"Yes: {result.yes_stake:,.0f} NEURO ({result.approval_rate:.1%})")
print(f"No: {result.no_stake:,.0f} NEURO")
print(f"Participation: {result.participation_rate:.1%}")
print(f"Quorum reached: {result.quorum_reached}")
print(f"Approved: {result.approved}")
```

### Vote History

```python
# See your voting history
votes = voting.get_voter_history(node.node_id)
for vote in votes:
    print(f"{vote.nep_id}: {vote.choice.value} ({vote.stake_at_vote} NEURO)")
```

## Voting Timeline

```
Day 0          Day 7           Day 14
  │              │                │
  │   REVIEW     │    VOTING      │    RESULT
  │              │                │
  ▼              ▼                ▼
┌─────────────┬─────────────────┬───────────────┐
│   Draft     │   Voting Open   │   Finalized   │
│   Posted    │   Cast Votes    │   Approved or │
│   Discussion│   See Tally     │   Rejected    │
└─────────────┴─────────────────┴───────────────┘
```

## Important Rules

### 1. One Vote Per Node

Each node can vote once per proposal. To change your vote, you must wait for the next proposal.

### 2. Stake Locked at Vote Time

Your voting power is determined by your stake **when you vote**, not when voting ends. This prevents last-minute stake manipulation.

### 3. No Delegation (Yet)

Currently, you cannot delegate your voting power. This may be added in a future NEP.

### 4. Votes Are Public

All votes are recorded on the ledger and visible to everyone. There is no anonymous voting.

## After Voting

### If Approved

1. **Grace Period** begins (typically 7-14 days)
2. **Activation Block** is scheduled
3. Nodes should upgrade before activation
4. At activation, new parameters take effect

### If Rejected

1. Proposal is marked REJECTED
2. Author can resubmit after 30 days
3. Must address feedback from community

## Validator Responsibilities

If you're a validator (100+ NEURO staked), you have additional responsibilities:

| Duty | Description |
|------|-------------|
| **Review** | Evaluate technical feasibility |
| **Vote** | Participate in all votes |
| **Upgrade** | Implement approved changes promptly |

Validators who consistently vote against consensus may face reputation penalties.

## FAQ

### Q: What if I don't vote?

Your stake doesn't count toward the result. The outcome is determined only by those who vote.

### Q: Can I vote on my own proposal?

Yes, but your vote is just like any other. You don't get extra weight for being the author.

### Q: What if quorum isn't reached?

The proposal fails. It can be resubmitted with more community outreach.

### Q: Can I see who voted for what?

Yes, all votes are public. Go to the proposal page to see the vote breakdown.

### Q: What happens during a tie?

A 50-50 tie fails to reach the 66% approval threshold, so the proposal is rejected.
