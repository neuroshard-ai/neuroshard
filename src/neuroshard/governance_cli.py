#!/usr/bin/env python3
"""
NeuroShard Governance CLI

Commands for participating in decentralized governance:
- List proposals
- View proposal details
- Create new proposals
- Vote on proposals
- Check voting results

Authentication:
- Uses wallet token (same as neuroshard-node)
- Token can be 64-char hex or 12-word mnemonic
- Derives ECDSA keypair for signing proposals/votes
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from neuroshard.version import __version__


# Lazy imports to speed up CLI startup
def get_governance_modules():
    """Import governance modules lazily."""
    from neuroshard.core.governance import (
        NEP, NEPType, NEPStatus, create_proposal, EconomicImpact,
        NEPRegistry, get_active_neps, get_pending_neps,
        cast_vote, get_vote_tally, VoteResult,
    )
    from neuroshard.core.governance.voting import VoteChoice, VotingSystem
    return {
        'NEP': NEP,
        'NEPType': NEPType,
        'NEPStatus': NEPStatus,
        'create_proposal': create_proposal,
        'EconomicImpact': EconomicImpact,
        'NEPRegistry': NEPRegistry,
        'get_active_neps': get_active_neps,
        'get_pending_neps': get_pending_neps,
        'cast_vote': cast_vote,
        'get_vote_tally': get_vote_tally,
        'VoteResult': VoteResult,
        'VoteChoice': VoteChoice,
        'VotingSystem': VotingSystem,
    }


def get_crypto_module():
    """Import crypto module lazily."""
    from neuroshard.core.crypto.ecdsa import NodeCrypto, derive_keypair_from_token
    return {
        'NodeCrypto': NodeCrypto,
        'derive_keypair_from_token': derive_keypair_from_token,
    }


NEUROSHARD_DIR = os.path.expanduser("~/.neuroshard")
TOKEN_FILE = os.path.join(NEUROSHARD_DIR, "wallet_token")


def resolve_token(token_arg: str = None) -> str:
    """
    Resolve wallet token from various sources (in priority order):
    1. --token argument
    2. NEUROSHARD_TOKEN environment variable
    3. ~/.neuroshard/wallet_token file
    
    Handles both 64-char hex and 12-word mnemonic formats.
    
    Returns the resolved token or None.
    """
    token = None
    source = None
    
    # Priority 1: Command line argument
    if token_arg:
        token = token_arg
        source = "command line"
    
    # Priority 2: Environment variable
    elif os.getenv("NEUROSHARD_TOKEN"):
        token = os.getenv("NEUROSHARD_TOKEN")
        source = "NEUROSHARD_TOKEN env"
    
    # Priority 3: Token file
    elif os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'r') as f:
                token = f.read().strip()
                source = TOKEN_FILE
        except Exception:
            pass
    
    if not token:
        return None, None
    
    # Handle mnemonic (12 words)
    words = token.strip().split()
    if len(words) == 12:
        try:
            from mnemonic import Mnemonic
            mnemo = Mnemonic("english")
            if mnemo.check(token):
                seed = mnemo.to_seed(token, passphrase="")
                token = seed[:32].hex()
                source = f"{source} (mnemonic)"
        except ImportError:
            print("[WARNING] 'mnemonic' package not installed, treating as raw token")
        except Exception as e:
            print(f"[WARNING] Mnemonic error: {e}")
    
    return token, source


def get_wallet(token: str):
    """
    Initialize wallet from token.
    
    Returns (node_id, crypto) tuple where:
    - node_id: Your unique identifier (derived from public key)
    - crypto: NodeCrypto instance for signing
    """
    crypto_mod = get_crypto_module()
    crypto = crypto_mod['NodeCrypto'](token)
    return crypto.node_id, crypto


def save_token(token: str):
    """Save token to file for future use."""
    os.makedirs(NEUROSHARD_DIR, exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    os.chmod(TOKEN_FILE, 0o600)  # Secure permissions
    print(f"[INFO] Token saved to {TOKEN_FILE}")


def get_registry(ledger=None):
    """Get or create the NEP registry."""
    gov = get_governance_modules()
    db_path = os.path.join(NEUROSHARD_DIR, "nep_registry.db")
    os.makedirs(NEUROSHARD_DIR, exist_ok=True)
    return gov['NEPRegistry'](db_path=db_path, ledger=ledger)


def get_voting_system(ledger=None):
    """Get or create the voting system."""
    gov = get_governance_modules()
    db_path = os.path.join(NEUROSHARD_DIR, "nep_votes.db")
    os.makedirs(NEUROSHARD_DIR, exist_ok=True)
    return gov['VotingSystem'](db_path=db_path, ledger=ledger)


def require_wallet(args) -> tuple:
    """
    Ensure wallet is available for commands that need it.
    
    Returns (node_id, crypto) or exits with error.
    """
    token, source = resolve_token(getattr(args, 'token', None))
    
    if not token:
        print("\n❌ Wallet token required for this command!")
        print()
        print("Provide your token using one of:")
        print("  1. --token YOUR_TOKEN")
        print("  2. export NEUROSHARD_TOKEN=YOUR_TOKEN")
        print(f"  3. echo YOUR_TOKEN > {TOKEN_FILE}")
        print()
        print("Get your token at: https://neuroshard.com/wallet")
        sys.exit(1)
    
    try:
        node_id, crypto = get_wallet(token)
        print(f"[WALLET] Authenticated as {node_id[:16]}... (from {source})")
        return node_id, crypto
    except Exception as e:
        print(f"\n❌ Failed to initialize wallet: {e}")
        sys.exit(1)


def format_timestamp(ts):
    """Format unix timestamp to readable string."""
    if not ts:
        return "N/A"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def format_status(status):
    """Format status with color codes."""
    colors = {
        'draft': '\033[90m',      # Gray
        'review': '\033[94m',     # Blue
        'voting': '\033[93m',     # Yellow
        'approved': '\033[92m',   # Green
        'rejected': '\033[91m',   # Red
        'scheduled': '\033[95m',  # Purple
        'active': '\033[92m',     # Green
    }
    reset = '\033[0m'
    color = colors.get(status, '')
    return f"{color}{status.upper()}{reset}"


def cmd_list(args):
    """List all proposals."""
    gov = get_governance_modules()
    registry = get_registry()
    
    # Filter by status if specified
    status_filter = None
    if args.status:
        try:
            status_filter = gov['NEPStatus'](args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid statuses: {[s.value for s in gov['NEPStatus']]}")
            return 1
    
    proposals = registry.list_proposals(status=status_filter, limit=args.limit)
    
    if not proposals:
        print("No proposals found.")
        return 0
    
    print(f"\n{'ID':<10} {'Status':<12} {'Type':<8} {'Title':<40} {'Created':<16}")
    print("=" * 90)
    
    for nep in proposals:
        print(
            f"{nep.nep_id:<10} "
            f"{format_status(nep.status.value):<22} "  # Extra space for color codes
            f"{nep.nep_type.value:<8} "
            f"{nep.title[:38]:<40} "
            f"{format_timestamp(nep.created_at):<16}"
        )
    
    print(f"\nTotal: {len(proposals)} proposals")
    return 0


def cmd_show(args):
    """Show details of a specific proposal."""
    gov = get_governance_modules()
    registry = get_registry()
    voting = get_voting_system()
    
    nep = registry.get_proposal(args.nep_id)
    if not nep:
        print(f"Proposal not found: {args.nep_id}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"  {nep.nep_id}: {nep.title}")
    print(f"{'='*70}")
    print()
    print(f"  Status:      {format_status(nep.status.value)}")
    print(f"  Type:        {nep.nep_type.value}")
    print(f"  Author:      {nep.author_node_id[:20]}...")
    print(f"  Created:     {format_timestamp(nep.created_at)}")
    print(f"  Updated:     {format_timestamp(nep.updated_at)}")
    print()
    print(f"  ABSTRACT")
    print(f"  {'-'*60}")
    print(f"  {nep.abstract}")
    print()
    
    # Economic impact
    impact = nep.economic_impact
    print(f"  ECONOMIC IMPACT")
    print(f"  {'-'*60}")
    print(f"  Training efficiency:  {impact.training_efficiency_multiplier:.1f}x")
    print(f"  Training reward:      {impact.training_reward_multiplier:.1f}x")
    print(f"  Inference reward:     {impact.inference_reward_multiplier:.1f}x")
    print(f"  Memory change:        {impact.min_memory_change_mb:+d} MB")
    print(f"  Net earnings:         {impact.net_earnings_change_percent:+.1f}%")
    print()
    
    # Voting info
    if nep.status == gov['NEPStatus'].VOTING:
        print(f"  VOTING (Active)")
        print(f"  {'-'*60}")
        print(f"  Started:   {format_timestamp(nep.voting_start)}")
        print(f"  Ends:      {format_timestamp(nep.voting_end)}")
        
        # Get tally
        result = voting.get_vote_tally(nep.nep_id, registry)
        print(f"  Yes votes: {result.yes_count} ({result.yes_stake:.2f} NEURO)")
        print(f"  No votes:  {result.no_count} ({result.no_stake:.2f} NEURO)")
        print(f"  Abstain:   {result.abstain_count} ({result.abstain_stake:.2f} NEURO)")
        print(f"  Quorum:    {result.participation_rate*100:.1f}% of {result.total_network_stake:.2f} NEURO")
        if result.quorum_reached:
            print(f"  Approval:  {result.approval_rate*100:.1f}% (threshold: 66%)")
        print()
    
    if args.full:
        print(f"  MOTIVATION")
        print(f"  {'-'*60}")
        print(f"  {nep.motivation}")
        print()
        print(f"  SPECIFICATION")
        print(f"  {'-'*60}")
        print(f"  {nep.specification}")
        print()
    
    return 0


def cmd_propose(args):
    """Create a new proposal."""
    gov = get_governance_modules()
    
    # Require wallet authentication
    node_id, crypto = require_wallet(args)
    
    registry = get_registry()
    
    # Parse type
    try:
        nep_type = gov['NEPType'](args.type)
    except ValueError:
        print(f"Invalid type: {args.type}")
        print(f"Valid types: {[t.value for t in gov['NEPType']]}")
        return 1
    
    # If using a file for content
    if args.file:
        with open(args.file) as f:
            content = json.load(f)
            title = content.get('title', args.title)
            abstract = content.get('abstract', '')
            motivation = content.get('motivation', '')
            specification = content.get('specification', '')
    else:
        title = args.title
        abstract = args.abstract or input("Enter abstract (1-2 sentences): ")
        motivation = args.motivation or input("Enter motivation (why is this needed?): ")
        specification = args.specification or "See discussion thread."
    
    # Create proposal
    nep = gov['create_proposal'](
        title=title,
        nep_type=nep_type,
        abstract=abstract,
        motivation=motivation,
        specification=specification,
        author_node_id=node_id,
        economic_impact=gov['EconomicImpact'](
            net_earnings_change_percent=args.earnings_impact or 0.0
        ),
    )
    
    # Confirm submission
    print(f"\n{'='*60}")
    print(f"  PROPOSAL PREVIEW")
    print(f"{'='*60}")
    print(f"  Title:    {title}")
    print(f"  Type:     {nep_type.value}")
    print(f"  Author:   {node_id[:16]}...")
    print(f"  Abstract: {abstract[:100]}...")
    print()
    print(f"  ⚠️  COSTS:")
    print(f"     - 10 NEURO proposal fee (held in escrow)")
    print(f"     - Requires 100 NEURO staked")
    print()
    print(f"  💰 REWARDS (stake-proportional):")
    print(f"     - If approved: 0.1% of total stake voted + fee refund")
    print(f"     - If rejected w/ quorum: 0.01% of total stake voted")
    print(f"     - Example: 100K stake votes → ~110 NEURO if approved")
    print()
    
    if not args.yes:
        confirm = input("Submit this proposal? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Sign the proposal with ECDSA
    signature = crypto.sign(nep.content_hash)
    
    # Submit to registry
    success, message, nep_id = registry.submit_proposal(
        nep=nep,
        node_id=node_id,
        signature=signature,
    )
    
    if success:
        print(f"\n✅ {message}")
        print(f"   Your proposal ID: {nep_id}")
        print(f"   Your wallet: {node_id}")
        print(f"   Track status: neuroshard-governance show {nep_id}")
    else:
        print(f"\n❌ Failed: {message}")
        return 1
    
    return 0


def cmd_vote(args):
    """Vote on a proposal."""
    gov = get_governance_modules()
    
    # Require wallet authentication
    node_id, crypto = require_wallet(args)
    
    registry = get_registry()
    voting = get_voting_system()
    
    # Check proposal exists and is in voting
    nep = registry.get_proposal(args.nep_id)
    if not nep:
        print(f"Proposal not found: {args.nep_id}")
        return 1
    
    if nep.status != gov['NEPStatus'].VOTING:
        print(f"Proposal is not in voting phase (status: {nep.status.value})")
        return 1
    
    # Parse vote choice
    choice_map = {
        'yes': gov['VoteChoice'].YES,
        'no': gov['VoteChoice'].NO,
        'abstain': gov['VoteChoice'].ABSTAIN,
    }
    choice = choice_map.get(args.choice.lower())
    if not choice:
        print(f"Invalid vote: {args.choice}")
        print("Valid choices: yes, no, abstain")
        return 1
    
    # Show proposal summary
    print(f"\n{'='*60}")
    print(f"  VOTING: {nep.nep_id}")
    print(f"{'='*60}")
    print(f"  Title:  {nep.title}")
    print(f"  Impact: {nep.economic_impact.describe()}")
    print(f"  Ends:   {format_timestamp(nep.voting_end)}")
    print()
    print(f"  Your wallet: {node_id[:16]}...")
    print(f"  Your vote: {args.choice.upper()}")
    if args.reason:
        print(f"  Reason: {args.reason}")
    print()
    print(f"  💰 You will earn: 0.01% of your staked NEURO for voting")
    print()
    
    if not args.yes:
        confirm = input("Confirm vote? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Sign the vote with ECDSA
    vote_payload = f"{args.nep_id}:{node_id}:{choice.value}:{int(time.time())}"
    signature = crypto.sign(vote_payload)
    
    # Cast vote
    success, message = voting.cast_vote(
        nep_id=args.nep_id,
        voter_node_id=node_id,
        choice=choice,
        signature=signature,
        reason=args.reason or "",
        nep_registry=registry,
    )
    
    if success:
        print(f"\n✅ {message}")
        print(f"   Your vote has been recorded and signed.")
    else:
        print(f"\n❌ Failed: {message}")
        return 1
    
    return 0


def cmd_results(args):
    """Show voting results for a proposal."""
    gov = get_governance_modules()
    registry = get_registry()
    voting = get_voting_system()
    
    nep = registry.get_proposal(args.nep_id)
    if not nep:
        print(f"Proposal not found: {args.nep_id}")
        return 1
    
    result = voting.get_vote_tally(args.nep_id, registry)
    
    print(f"\n{'='*60}")
    print(f"  VOTING RESULTS: {nep.nep_id}")
    print(f"{'='*60}")
    print(f"  {nep.title}")
    print(f"  Status: {format_status(nep.status.value)}")
    print()
    
    # Vote counts
    total_votes = result.yes_count + result.no_count + result.abstain_count
    print(f"  VOTES ({total_votes} total)")
    print(f"  {'-'*50}")
    
    # Bar chart
    if total_votes > 0:
        yes_pct = result.yes_count / total_votes * 100
        no_pct = result.no_count / total_votes * 100
        abs_pct = result.abstain_count / total_votes * 100
        
        bar_width = 30
        yes_bar = '█' * int(yes_pct / 100 * bar_width)
        no_bar = '█' * int(no_pct / 100 * bar_width)
        
        print(f"  YES:     {result.yes_count:4} ({yes_pct:5.1f}%) \033[92m{yes_bar}\033[0m")
        print(f"  NO:      {result.no_count:4} ({no_pct:5.1f}%) \033[91m{no_bar}\033[0m")
        print(f"  ABSTAIN: {result.abstain_count:4} ({abs_pct:5.1f}%)")
    else:
        print(f"  No votes cast yet")
    print()
    
    # Stake-weighted results
    print(f"  STAKE-WEIGHTED")
    print(f"  {'-'*50}")
    print(f"  Yes stake:     {result.yes_stake:12.2f} NEURO")
    print(f"  No stake:      {result.no_stake:12.2f} NEURO")
    print(f"  Abstain stake: {result.abstain_stake:12.2f} NEURO")
    print(f"  Total voted:   {result.total_stake_voted:12.2f} NEURO")
    print(f"  Network stake: {result.total_network_stake:12.2f} NEURO")
    print()
    
    # Thresholds
    print(f"  THRESHOLDS")
    print(f"  {'-'*50}")
    quorum_status = "✅" if result.quorum_reached else "❌"
    print(f"  Quorum (20%):   {result.participation_rate*100:5.1f}% {quorum_status}")
    
    approval_status = "✅" if result.approval_rate >= 0.66 else "❌"
    print(f"  Approval (66%): {result.approval_rate*100:5.1f}% {approval_status}")
    print()
    
    if result.quorum_reached and result.approval_rate >= 0.66:
        print(f"  \033[92m✓ Proposal PASSING\033[0m")
    elif result.quorum_reached:
        print(f"  \033[91m✗ Proposal FAILING (below approval threshold)\033[0m")
    else:
        print(f"  \033[93m⏳ Waiting for quorum\033[0m")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroShard Governance CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WALLET SETUP:
  Commands that modify state (propose, vote) require your wallet token.
  This is the same token used for neuroshard-node.

  Option 1: Pass directly
    neuroshard-governance --token YOUR_TOKEN propose ...

  Option 2: Environment variable
    export NEUROSHARD_TOKEN=YOUR_TOKEN
    neuroshard-governance propose ...

  Option 3: Save to file (recommended)
    neuroshard-governance --token YOUR_TOKEN wallet --save
    # Future commands auto-use ~/.neuroshard/wallet_token

EXAMPLES:
  # Check your wallet
  neuroshard-governance --token YOUR_TOKEN wallet

  # List all proposals (no auth needed)
  neuroshard-governance list

  # List proposals in voting
  neuroshard-governance list --status voting

  # View a proposal
  neuroshard-governance show NEP-001

  # Create a proposal (requires wallet)
  neuroshard-governance propose --title "Add MTP Training" --type train

  # Vote on a proposal (requires wallet)
  neuroshard-governance vote NEP-001 yes --reason "Improves efficiency"

  # Check voting results
  neuroshard-governance results NEP-001

Get your wallet token at: https://neuroshard.com/wallet
        """
    )
    
    parser.add_argument(
        "--version", action="version",
        version=f"NeuroShard Governance CLI {__version__}"
    )
    
    # Global token option (for commands that need authentication)
    parser.add_argument(
        "--token", type=str, default=None,
        help="Wallet token (64-char hex or 12-word mnemonic). "
             "Can also be set via NEUROSHARD_TOKEN env or ~/.neuroshard/wallet_token"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command (no auth required)
    list_parser = subparsers.add_parser("list", help="List proposals")
    list_parser.add_argument("--status", type=str, help="Filter by status")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results")
    
    # Show command (no auth required)
    show_parser = subparsers.add_parser("show", help="Show proposal details")
    show_parser.add_argument("nep_id", type=str, help="Proposal ID (e.g., NEP-001)")
    show_parser.add_argument("--full", action="store_true", help="Show full specification")
    
    # Propose command (requires wallet)
    propose_parser = subparsers.add_parser("propose", help="Create a proposal (requires wallet)")
    propose_parser.add_argument("--title", type=str, required=True, help="Proposal title")
    propose_parser.add_argument("--type", type=str, required=True, 
                                help="Type: arch, econ, train, net, gov, emergency")
    propose_parser.add_argument("--abstract", type=str, help="Brief description")
    propose_parser.add_argument("--motivation", type=str, help="Why is this needed")
    propose_parser.add_argument("--specification", type=str, help="Technical details")
    propose_parser.add_argument("--file", type=str, help="JSON file with proposal content")
    propose_parser.add_argument("--earnings-impact", type=float, help="Net earnings change %")
    propose_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Vote command (requires wallet)
    vote_parser = subparsers.add_parser("vote", help="Vote on a proposal (requires wallet)")
    vote_parser.add_argument("nep_id", type=str, help="Proposal ID")
    vote_parser.add_argument("choice", type=str, help="Vote: yes, no, abstain")
    vote_parser.add_argument("--reason", type=str, help="Reason for your vote")
    vote_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Results command (no auth required)
    results_parser = subparsers.add_parser("results", help="Show voting results")
    results_parser.add_argument("nep_id", type=str, help="Proposal ID")
    
    # Wallet command (utility)
    wallet_parser = subparsers.add_parser("wallet", help="Show wallet info")
    wallet_parser.add_argument("--save", action="store_true", 
                               help="Save token to ~/.neuroshard/wallet_token")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    commands = {
        'list': cmd_list,
        'show': cmd_show,
        'propose': cmd_propose,
        'vote': cmd_vote,
        'results': cmd_results,
        'wallet': cmd_wallet,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


def cmd_wallet(args):
    """Show wallet info and optionally save token."""
    token, source = resolve_token(args.token)
    
    if not token:
        print("\n❌ No wallet token found!")
        print()
        print("Provide your token using one of:")
        print("  1. --token YOUR_TOKEN")
        print("  2. export NEUROSHARD_TOKEN=YOUR_TOKEN")
        print(f"  3. echo YOUR_TOKEN > {TOKEN_FILE}")
        print()
        print("Get your token at: https://neuroshard.com/wallet")
        return 1
    
    try:
        node_id, crypto = get_wallet(token)
        
        print(f"\n{'='*60}")
        print(f"  WALLET INFO")
        print(f"{'='*60}")
        print(f"  Node ID:    {node_id}")
        print(f"  Public Key: {crypto.get_public_key_hex()[:32]}...")
        print(f"  Source:     {source}")
        print()
        
        # Test signing
        test_msg = "test_message"
        sig = crypto.sign(test_msg)
        verified = crypto.verify(test_msg, sig)
        print(f"  Signature test: {'✅ OK' if verified else '❌ FAILED'}")
        print()
        
        if args.save:
            save_token(token)
            print(f"  ✅ Token saved to {TOKEN_FILE}")
            print(f"     Future commands will use this automatically.")
        else:
            print(f"  💡 Run with --save to remember this wallet")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed to initialize wallet: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
