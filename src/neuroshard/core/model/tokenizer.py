"""
NeuroLLM Tokenizer — The Ever-Growing Tokenizer

A BPE (Byte Pair Encoding) tokenizer for NeuroLLM that is trained from scratch
by the network itself. Unlike GPT-4 or LLaMA which freeze their vocabulary
at training time, NeuroShard's tokenizer continuously learns new subword units
as more training data flows through the network.

Architecture:
- Byte-level BPE with GPT-4-class pre-tokenization regex
- Starts with 266 base tokens (10 special + 256 bytes)
- Learns merges from contributed training data (ever-growing)
- Distributed via CDN, auto-synced every 10 minutes
- Model embeddings auto-expand in 32K chunks to accommodate growth
- All old data remains compatible (merges only add, never remove)

Pre-tokenization:
- Uses a regex pattern derived from GPT-4/cl100k_base that properly handles:
  - Contractions (don't -> [don, 't])
  - Numbers split into 1-3 digit groups (123456 -> [123, 456])
  - Unicode letters and numbers as separate classes
  - Whitespace handling with newline awareness
  - Punctuation isolated from words
- This is critical for BPE quality — without good pre-tokenization,
  BPE learns cross-word merges that hurt generalization.
"""

import os
import json
import logging
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# GPT-4-CLASS PRE-TOKENIZATION PATTERN
# =============================================================================
# This regex splits text into meaningful chunks BEFORE BPE is applied.
# It's the single most important factor for tokenizer quality.
#
# Pattern breakdown:
#   1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive)
#   2. Unicode letters (optionally preceded by a non-letter/non-number)
#   3. Numbers in 1-3 digit groups (prevents merging "123456" into one token)
#   4. Non-letter, non-number, non-whitespace characters (punctuation)
#   5. Whitespace sequences (with newline awareness)
#
# Why this matters: Without pre-tokenization, BPE will merge across word
# boundaries (e.g., "the" + " " + "cat" -> "the cat"), creating tokens that
# don't generalize. Pre-tokenization ensures BPE only merges WITHIN words.
#
# Compatible with the `regex` package for full Unicode support, but falls back
# to a simpler `re`-based pattern if `regex` is unavailable.
# =============================================================================

try:
    import regex
    # Full GPT-4/cl100k_base-class pattern with Unicode property support
    _PRETOKENIZE_PATTERN = regex.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    _USE_REGEX = True
except ImportError:
    # Fallback: re-based pattern (no Unicode property classes, but close)
    _PRETOKENIZE_PATTERN = re.compile(
        r"""'(?:[sdtm]|ll|ve|re)|[a-zA-Z]+|[0-9]{1,3}| ?[^\s\w][\r\n]*|\s*[\r\n]|\s+"""
    )
    _USE_REGEX = False


class NeuroTokenizer:
    """
    A truly decentralized BPE tokenizer for NeuroLLM.
    
    Unlike traditional tokenizers that are pre-trained on massive corpora,
    this tokenizer starts with a minimal vocabulary and learns from the
    training data contributed by network participants.
    """
    
    # Special tokens (reserved IDs 0-9)
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"
    
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    
    # Byte tokens start at ID 10 (256 bytes = IDs 10-265)
    BYTE_OFFSET = 10
    
    # Learned merges start at ID 266
    MERGE_OFFSET = 266
    
    def __init__(self, vocab_size: int = 10_000_000):  # 10M - effectively unlimited
        """
        Initialize the NeuroLLM tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size. Default 10M is effectively unlimited.
                       The tokenizer can grow as large as needed - memory is the only
                       real constraint. For reference: GPT-4 ~100K, most LLMs ~32K-256K.
        """
        self.vocab_size = vocab_size
        
        # Core vocabulary
        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        
        # Byte vocabulary (256 bytes)
        self.byte_to_id = {i: i + self.BYTE_OFFSET for i in range(256)}
        self.id_to_byte = {v: k for k, v in self.byte_to_id.items()}
        
        # Learned BPE merges: (token1, token2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_to_tokens: Dict[int, Tuple[int, int]] = {}  # Reverse lookup
        
        # Next available ID for new merges
        self.next_merge_id = self.MERGE_OFFSET
        
        # Track which data sources have contributed merges
        # Format: {"source_name": num_merges_contributed}
        self.sources_contributed: Dict[str, int] = {}
        
        # Statistics
        self.total_tokens_processed = 0
        
        # Word-level cache for fast encoding (massive speedup for repeated words)
        # Key: word (str), Value: token_ids (tuple of ints)
        self._word_cache: Dict[str, Tuple[int, ...]] = {}
        self._word_cache_max_size = 500_000  # Limit cache size
        self._word_cache_hits = 0
        self._word_cache_misses = 0
        
        # Thread-safe cache access
        import threading
        self._cache_lock = threading.Lock()
        
        # Pre-tokenization pattern (GPT-4-class)
        # This splits text into linguistically meaningful chunks BEFORE BPE.
        # Using the module-level pattern for consistency.
        self._chunk_pattern = _PRETOKENIZE_PATTERN
        
        logger.info(f"NeuroTokenizer initialized with vocab_size={vocab_size}, "
                     f"pre-tokenization={'regex (full Unicode)' if _USE_REGEX else 're (ASCII fallback)'}")
    
    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID
    
    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID
    
    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID
    
    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID
    
    @property
    def current_vocab_size(self) -> int:
        """
        The current vocabulary size (valid token IDs: 0 to current_vocab_size-1).
        
        This grows as the tokenizer learns BPE merges:
        - Initial: 266 (10 special + 256 bytes)
        - After learning: 266 + num_merges
        - Maximum: vocab_size (10M default - effectively unlimited)
        
        IMPORTANT: During inference, only tokens 0 to current_vocab_size-1 are valid.
        Tokens beyond this have no learned representation and should not be sampled.
        """
        return self.next_merge_id
    
    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to byte-level token IDs.
        
        Applies Unicode NFC normalization first to ensure consistent byte
        representations (e.g., e + combining accent -> single accented char).
        """
        normalized = unicodedata.normalize('NFC', text)
        return [self.byte_to_id[b] for b in normalized.encode('utf-8')]
    
    def _apply_merges(self, token_ids: List[int]) -> List[int]:
        """
        Apply learned BPE merges to a sequence of token IDs.
        
        OPTIMIZED: Uses heap-based approach for O(n log n) instead of O(n²).
        Merges are applied in priority order (lower merge ID = higher priority).
        """
        if not self.merges or len(token_ids) <= 1:
            return token_ids
        
        import heapq
        
        # Convert to list for in-place modification
        tokens = list(token_ids)
        n = len(tokens)
        
        # Track which positions are "deleted" (merged into previous)
        deleted = [False] * n
        
        # Build initial heap of mergeable pairs: (merge_id, position)
        # Lower merge_id = higher priority (learned earlier = more frequent)
        heap = []
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                heapq.heappush(heap, (self.merges[pair], i))
        
        while heap:
            merge_id, pos = heapq.heappop(heap)
            
            # Skip if position was already processed
            if pos >= n - 1 or deleted[pos]:
                continue
            
            # Find actual next non-deleted position
            next_pos = pos + 1
            while next_pos < n and deleted[next_pos]:
                next_pos += 1
            
            if next_pos >= n:
                continue
            
            # Check if this merge still applies
            pair = (tokens[pos], tokens[next_pos])
            if pair not in self.merges or self.merges[pair] != merge_id:
                continue
            
            # Apply merge: replace token at pos, mark next_pos as deleted
            tokens[pos] = merge_id
            deleted[next_pos] = True
            
            # Find previous non-deleted position
            prev_pos = pos - 1
            while prev_pos >= 0 and deleted[prev_pos]:
                prev_pos -= 1
            
            # Find next-next non-deleted position  
            next_next_pos = next_pos + 1
            while next_next_pos < n and deleted[next_next_pos]:
                next_next_pos += 1
            
            # Add new potential merges to heap
            if prev_pos >= 0:
                new_pair = (tokens[prev_pos], tokens[pos])
                if new_pair in self.merges:
                    heapq.heappush(heap, (self.merges[new_pair], prev_pos))
            
            if next_next_pos < n:
                new_pair = (tokens[pos], tokens[next_next_pos])
                if new_pair in self.merges:
                    heapq.heappush(heap, (self.merges[new_pair], pos))
        
        # Build result excluding deleted positions
        return [tokens[i] for i in range(n) if not deleted[i]]
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        OPTIMIZED: Uses word-level caching for 5-20x speedup.
        Most words in text are repeated - cache their tokenizations.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length (truncate if longer)
            truncation: Whether to truncate
            padding: Whether to pad to max_length
            
        Returns:
            List of token IDs
        """
        # Use word-level caching for speed
        token_ids = self._encode_with_cache(text)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.BOS_ID] + token_ids + [self.EOS_ID]
        
        # Truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Padding
        if padding and max_length and len(token_ids) < max_length:
            token_ids = token_ids + [self.PAD_ID] * (max_length - len(token_ids))
        
        self.total_tokens_processed += len(token_ids)
        return token_ids
    
    def _invalidate_cache(self):
        """Invalidate the word cache after merges change.
        
        Must be called whenever new merges are added (learn_merges, load, etc.)
        to prevent stale tokenizations from being served.
        """
        with self._cache_lock:
            self._word_cache.clear()
            logger.debug(f"Word cache invalidated (was {self._word_cache_hits} hits, "
                        f"{self._word_cache_misses} misses)")
    
    def _encode_with_cache(self, text: str) -> List[int]:
        """
        Encode text using chunk-level caching (thread-safe).
        
        Split text into linguistically meaningful chunks via the pre-tokenization
        regex, look up each in cache, only tokenize cache misses.
        This gives massive speedup for repeated text patterns.
        """
        # Split into linguistically meaningful chunks (GPT-4-class pre-tokenization)
        chunks = self._chunk_pattern.findall(text)
        
        all_tokens = []
        for chunk in chunks:
            if not chunk:  # Skip empty chunks
                continue
            
            # Thread-safe cache lookup
            with self._cache_lock:
                cached = self._word_cache.get(chunk)
            
            if cached is not None:
                self._word_cache_hits += 1
                all_tokens.extend(cached)
            else:
                # Cache miss - tokenize and store
                self._word_cache_misses += 1
                byte_ids = self._text_to_bytes(chunk)
                chunk_tokens = tuple(self._apply_merges(byte_ids))
                
                # Thread-safe cache store (with size limit)
                with self._cache_lock:
                    if len(self._word_cache) < self._word_cache_max_size:
                        self._word_cache[chunk] = chunk_tokens
                
                all_tokens.extend(chunk_tokens)
        
        return all_tokens
    
    def _decode_token(self, token_id: int) -> bytes:
        """Decode a single token ID to bytes."""
        # Special tokens
        if token_id in [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]:
            return b''
        
        # Byte token
        if token_id in self.id_to_byte:
            return bytes([self.id_to_byte[token_id]])
        
        # Merged token - recursively decode
        if token_id in self.merge_to_tokens:
            t1, t2 = self.merge_to_tokens[token_id]
            return self._decode_token(t1) + self._decode_token(t2)
        
        # Unknown token - this should NOT happen in normal operation
        # If we get here, the model output a token ID beyond current_vocab_size
        # This is a bug in the generation code (should be masking invalid tokens)
        logger.warning(f"Unknown token ID {token_id} (vocab_size={self.current_vocab_size}) - using UNK")
        return b'<unk>'
    
    def decode(
        self, 
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        byte_sequence = b''
        
        for tid in token_ids:
            if skip_special_tokens and tid in [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]:
                continue
            byte_sequence += self._decode_token(tid)
        
        # Decode UTF-8, replacing errors
        return byte_sequence.decode('utf-8', errors='replace')
    
    def learn_merges(self, texts: List[str], num_merges: int = 1000, min_frequency: int = 2):
        """
        Learn new BPE merges from training data using an optimized algorithm.
        
        This uses incremental pair counting with a heap for O(n log n) performance
        instead of the naive O(n^2 x m) algorithm.
        
        Pre-tokenization uses the GPT-4-class regex to split text into
        linguistically meaningful chunks before learning merges. This ensures
        BPE only merges within word boundaries, producing higher quality tokens.
        
        Args:
            texts: List of training texts
            num_merges: Number of new merges to learn
            min_frequency: Minimum pair frequency to create merge
        """
        import heapq
        
        if self.next_merge_id + num_merges > self.vocab_size:
            num_merges = self.vocab_size - self.next_merge_id
            if num_merges <= 0:
                logger.warning("Vocabulary is full, cannot learn more merges")
                return
        
        logger.info(f"Tokenizing {len(texts)} texts...")
        
        # Tokenize all texts to current vocabulary
        # Use GPT-4-class pre-tokenization to split into meaningful chunks first
        # This ensures BPE merges don't cross word boundaries
        word_freq: Counter = Counter()
        for text in texts:
            # Apply Unicode NFC normalization for consistent byte representation
            text = unicodedata.normalize('NFC', text)
            # Split using the pre-tokenization pattern (handles contractions,
            # numbers, punctuation, whitespace properly)
            chunks = self._chunk_pattern.findall(text)
            for chunk in chunks:
                if chunk.strip():  # Skip pure whitespace chunks
                    word_freq[chunk] += 1
        
        logger.info(f"Found {len(word_freq)} unique words")
        
        # Convert words to byte sequences with frequency
        # Format: {word_tuple: frequency} where word_tuple is tuple of token ids
        word_tokens: Dict[tuple, int] = {}
        for word, freq in word_freq.items():
            byte_ids = tuple(self._text_to_bytes(word))
            token_ids = tuple(self._apply_merges(list(byte_ids)))
            if token_ids in word_tokens:
                word_tokens[token_ids] += freq
            else:
                word_tokens[token_ids] = freq
        
        logger.info(f"Converted to {len(word_tokens)} unique token sequences")
        
        # Build initial pair counts
        pair_counts: Counter = Counter()
        # Track which words contain which pairs for efficient updates
        pair_to_words: Dict[Tuple[int, int], Set[tuple]] = {}
        
        for word, freq in word_tokens.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word)
        
        logger.info(f"Initial pair count: {len(pair_counts)} unique pairs")
        
        merges_learned = 0
        log_interval = max(1, num_merges // 20)  # Log ~20 times during learning
        
        while merges_learned < num_merges:
            if not pair_counts:
                logger.info("No more pairs to merge")
                break
            
            # Find most frequent pair
            best_pair, count = pair_counts.most_common(1)[0]
            
            if count < min_frequency:
                logger.info(f"Best pair frequency {count} below minimum {min_frequency}")
                break
            
            # Create new merge
            new_id = self.next_merge_id
            self.merges[best_pair] = new_id
            self.merge_to_tokens[new_id] = best_pair
            self.next_merge_id += 1
            merges_learned += 1
            
            if merges_learned % log_interval == 0:
                logger.info(f"  Learned {merges_learned}/{num_merges} merges, best pair freq={count}")
            
            # Update word_tokens and pair_counts incrementally
            words_to_update = pair_to_words.get(best_pair, set()).copy()
            
            # Remove the merged pair from counts
            del pair_counts[best_pair]
            if best_pair in pair_to_words:
                del pair_to_words[best_pair]
            
            for old_word in words_to_update:
                if old_word not in word_tokens:
                    continue
                    
                freq = word_tokens[old_word]
                
                # Remove old pair counts for this word
                for i in range(len(old_word) - 1):
                    pair = (old_word[i], old_word[i + 1])
                    if pair in pair_counts:
                        pair_counts[pair] -= freq
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                        if pair in pair_to_words and old_word in pair_to_words[pair]:
                            pair_to_words[pair].discard(old_word)
                
                # Apply merge to create new word
                new_word = []
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and (old_word[i], old_word[i + 1]) == best_pair:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(old_word[i])
                        i += 1
                new_word = tuple(new_word)
                
                # Update word_tokens
                del word_tokens[old_word]
                if new_word in word_tokens:
                    word_tokens[new_word] += freq
                else:
                    word_tokens[new_word] = freq
                
                # Add new pair counts for this word
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] += freq
                    if pair not in pair_to_words:
                        pair_to_words[pair] = set()
                    pair_to_words[pair].add(new_word)
        
        # Invalidate word cache — all cached tokenizations are now stale
        self._invalidate_cache()
        
        logger.info(f"Learned {merges_learned} new merges, vocab size now {len(self)}")
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        input_ids = []
        attention_mask = []
        
        for text in texts:
            ids = self.encode(text, max_length=max_length, truncation=truncation, padding=padding)
            input_ids.append(ids)
            attention_mask.append([1 if tid != self.PAD_ID else 0 for tid in ids])
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def save(self, path: str):
        """
        Save tokenizer to a JSON file.
        
        Args:
            path: Path to save the tokenizer JSON file (must end with .json)
        """
        if not path.endswith('.json'):
            path = path + '.json'
        
        config = {
            "vocab_size": self.vocab_size,
            "next_merge_id": self.next_merge_id,
            "total_tokens_processed": self.total_tokens_processed,
            "sources_contributed": self.sources_contributed,
            # Convert tuple keys to strings for JSON
            "merges": {f"{k[0]}_{k[1]}": v for k, v in self.merges.items()},
        }
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Tokenizer saved to {path} ({self.current_vocab_size} tokens, {len(self.merges)} merges)")
    
    @classmethod
    def load(cls, path: str) -> 'NeuroTokenizer':
        """
        Load tokenizer from a JSON file.
        
        Args:
            path: Path to the tokenizer JSON file
            
        Returns:
            Loaded NeuroTokenizer instance
        """
        if not path.endswith('.json'):
            path = path + '.json'
        
        if not os.path.exists(path):
            logger.warning(f"No tokenizer found at {path}, creating new one")
            return cls()
        
        with open(path) as f:
            config = json.load(f)
        
        tokenizer = cls(vocab_size=config.get("vocab_size", 10_000_000))  # Default to unlimited
        tokenizer.next_merge_id = config.get("next_merge_id", cls.MERGE_OFFSET)
        tokenizer.total_tokens_processed = config.get("total_tokens_processed", 0)
        tokenizer.sources_contributed = config.get("sources_contributed", {})
        
        # Restore merges
        merges_data = config.get("merges", {})
        for key_str, merged_id in merges_data.items():
            t1, t2 = map(int, key_str.split("_"))
            tokenizer.merges[(t1, t2)] = merged_id
            tokenizer.merge_to_tokens[merged_id] = (t1, t2)
        
        # Invalidate cache (loaded merges may differ from any previously cached)
        tokenizer._invalidate_cache()
        
        logger.info(f"Tokenizer loaded from {path} ({tokenizer.current_vocab_size} tokens, {len(tokenizer.merges)} merges)")
        return tokenizer
    
    def __len__(self) -> int:
        """Return current vocabulary size (valid token count)."""
        # This should match current_vocab_size for consistency
        # Base: 266 tokens (IDs 0-265: 10 special/reserved + 256 bytes)
        # Plus: learned merges
        return self.current_vocab_size
    
    def has_source_contributed(self, source_name: str) -> bool:
        """Check if a data source has already contributed merges."""
        return source_name in self.sources_contributed
    
    def record_source_contribution(self, source_name: str, num_merges: int):
        """Record that a source has contributed merges."""
        self.sources_contributed[source_name] = num_merges
        logger.info(f"Recorded contribution: '{source_name}' added {num_merges} merges")
    
    def get_stats(self) -> Dict:
        """Get tokenizer statistics."""
        return {
            "vocab_size": self.vocab_size,
            "current_vocab": len(self),
            "num_merges": len(self.merges),
            "total_tokens_processed": self.total_tokens_processed,
            "can_learn_more": self.next_merge_id < self.vocab_size,
            "sources_contributed": self.sources_contributed,
        }


# Global tokenizer instance
_tokenizer: Optional[NeuroTokenizer] = None


def get_neuro_tokenizer() -> NeuroTokenizer:
    """Get the global NeuroLLM tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = NeuroTokenizer()
    return _tokenizer


def reset_tokenizer():
    """Reset the global tokenizer (for testing)."""
    global _tokenizer
    _tokenizer = None
