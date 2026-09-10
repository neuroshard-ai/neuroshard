#!/usr/bin/env python3
"""
NeuroShard Dynamic Architecture Migration Script

This script prepares the system for the new dynamic width + depth scaling.

WHAT IT DOES:
1. Deletes old checkpoints (incompatible tensor shapes)
2. Optionally resets ledger for clean launch
3. Preserves Genesis data (architecture-agnostic)
4. Creates version marker

WHY RESET?
- Old checkpoints hardcoded to 768 hidden_dim
- New system uses variable hidden_dim based on network size
- Tensor shapes are incompatible (cannot load)

WHAT'S PRESERVED:
- Genesis training data (works with any architecture)
- Tokenizer vocabulary (architecture-independent)
- Network topology (P2P connections)
"""

import os
import shutil
import sys
from pathlib import Path
import argparse


def migrate_to_dynamic_architecture(reset_ledger: bool = False, dry_run: bool = False):
    """
    Migrate to dynamic architecture system.
    
    Args:
        reset_ledger: If True, reset NEURO ledger (recommended for clean launch)
        dry_run: If True, show what would be done without actually doing it
    """
    print("=" * 70)
    print("NEUROSHARD DYNAMIC ARCHITECTURE MIGRATION")
    print("=" * 70)
    print()
    print("This script will prepare your system for dynamic width + depth scaling.")
    print()
    
    neuroshard_dir = Path.home() / ".neuroshard"
    neuroshard_dir.mkdir(parents=True, exist_ok=True)
    
    actions = []
    
    # 1. Checkpoint cleanup
    checkpoint_dir = neuroshard_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            actions.append({
                "action": "DELETE",
                "path": checkpoint_dir,
                "reason": "Incompatible tensor shapes (hardcoded 768-dim)",
                "count": len(checkpoint_files)
            })
    
    # 2. Ledger reset (optional)
    ledger_db = neuroshard_dir / "ledger.db"
    if ledger_db.exists() and reset_ledger:
        actions.append({
            "action": "DELETE",
            "path": ledger_db,
            "reason": "Clean launch with new reward structure",
            "count": 1
        })
    
    # 3. Data cache cleanup (optional - helps free space)
    data_cache_dir = checkpoint_dir / "data_cache"
    if data_cache_dir.exists():
        cache_size_mb = sum(f.stat().st_size for f in data_cache_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        if cache_size_mb > 100:  # Only mention if > 100MB
            actions.append({
                "action": "OPTIONAL",
                "path": data_cache_dir,
                "reason": f"Free {cache_size_mb:.0f}MB disk space (will re-download shards)",
                "count": len(list(data_cache_dir.rglob('*')))
            })
    
    # Display plan
    if not actions:
        print("✅ No migration needed! System is already clean.")
        return
    
    print("MIGRATION PLAN:")
    print()
    for i, action in enumerate(actions, 1):
        print(f"{i}. [{action['action']}] {action['path']}")
        print(f"   Reason: {action['reason']}")
        if action.get('count'):
            print(f"   Files: {action['count']}")
        print()
    
    print("PRESERVED (no changes):")
    print("  ✅ Genesis training data (architecture-agnostic)")
    print("  ✅ Tokenizer vocabulary (32k tokens)")
    print("  ✅ P2P network configuration")
    if not reset_ledger:
        print("  ✅ NEURO ledger and balances")
    print()
    
    if dry_run:
        print("DRY RUN - No actual changes made")
        return
    
    # Confirm
    print("⚠️  This will delete old model checkpoints.")
    if reset_ledger:
        print("⚠️  This will RESET your NEURO balance to zero!")
    print()
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled.")
        return
    
    # Execute migration
    print()
    print("Executing migration...")
    print()
    
    for action in actions:
        if action['action'] == 'DELETE':
            print(f"🗑️  Deleting: {action['path']}")
            try:
                if action['path'].is_dir():
                    shutil.rmtree(action['path'])
                    action['path'].mkdir(parents=True, exist_ok=True)
                else:
                    action['path'].unlink()
                print(f"   ✅ Deleted")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        elif action['action'] == 'OPTIONAL':
            response = input(f"Delete {action['path']}? (yes/no): ")
            if response.lower() == 'yes':
                shutil.rmtree(action['path'])
                print(f"   ✅ Deleted")
    
    # Create version marker
    version_file = neuroshard_dir / "version.txt"
    with open(version_file, "w") as f:
        f.write("2.0-dynamic\n")
        f.write("Migration completed\n")
        f.write(f"Timestamp: {Path(version_file).stat().st_mtime}\n")
    
    print()
    print("=" * 70)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Restart your NeuroShard node")
    print("2. Network will auto-detect capacity and set optimal architecture")
    print("3. Model will scale dynamically as more nodes join!")
    print()
    print("Architecture scaling examples:")
    print("  • 10 nodes (40GB):    ~300M params (12L × 1024H)")
    print("  • 100 nodes (800GB):  ~3.5B params (24L × 2048H)")
    print("  • 1000 nodes (8TB):   ~35B params (48L × 4096H)")
    print()
    print("No more fixed 768-dim bottleneck! 🚀")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NeuroShard to dynamic architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without executing
  python migrate_to_dynamic_architecture.py --dry-run
  
  # Migrate with ledger reset (recommended for clean launch)
  python migrate_to_dynamic_architecture.py --reset-ledger
  
  # Migrate preserving NEURO balances
  python migrate_to_dynamic_architecture.py
        """
    )
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="Reset NEURO ledger (recommended for clean launch)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    migrate_to_dynamic_architecture(
        reset_ledger=args.reset_ledger,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

