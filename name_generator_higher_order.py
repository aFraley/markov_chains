"""
=============================================================================
Higher-Order Markov Chain Name Generator — Python Edition
=============================================================================

WHAT'S DIFFERENT FROM ORDER-1?
------------------------------
The basic (order-1) chain asks: "Given the last 1 character, what comes next?"
A higher-order chain asks: "Given the last N characters, what comes next?"

This is a big quality leap. For example, with order-2:
  - After seeing "th", the chain knows "e", "o", "i", "a" are likely
  - Order-1 only sees "h" and has no idea it was preceded by "t"

THE SPARSITY PROBLEM AND BACKOFF:
----------------------------------
With ~45 training names, many 2-char or 3-char sequences will never appear.
We solve this with BACKOFF: try order-N first, if that context was never
seen, fall back to order-(N-1), down to order-1.

This is the "stupid backoff" of classic n-gram language models. Its smarter
descendants (Katz backoff, Kneser-Ney smoothing) reserve probability mass for
unseen contexts instead of just falling back.

Example with order-3, context "xqz":
  "xqz" -> not in chain -> try "qz" -> not in chain -> try "z" -> found!

Based on: name_generator.py (order-1 Markov chain)
=============================================================================
"""

import math
import random

from name_generator import NAME_SETS


# =============================================================================
# CHAIN CACHE — keyed by "type:order" since different orders produce
# different chains from the same training data.
# =============================================================================

_chain_cache: dict[str, dict] = {}


# =============================================================================
# CORE ALGORITHM: Building the Higher-Order Chain
# =============================================================================

def construct_chain(name_list: list[str], order: int) -> dict:
    """
    Build a Markov chain that records transitions for ALL orders from 1 to N.

    This means we can always fall back to a shorter context during generation.

    For order=2 and the name "Achilles":
      Order-1 transitions: A->c, c->h, h->i, i->l, l->l, l->e, e->s
      Order-2 transitions: Ac->h, ch->i, hi->l, il->l, ll->e, le->s

    For order=3 and the name "Achilles":
      Order-1: A->c, c->h, h->i, i->l, l->l, l->e, e->s
      Order-2: Ac->h, ch->i, hi->l, il->l, ll->e, le->s
      Order-3: Ach->i, chi->l, hil->l, ill->e, lle->s

    Initials also become multi-char:
      Order-1: "initial" = "A"
      Order-2: "initial_1" = "A", "initial_2" = "Ac"
      Order-3: "initial_1" = "A", "initial_2" = "Ac", "initial_3" = "Ach"

    Args:
        name_list: Training names to learn patterns from.
        order: The maximum context length (1, 2, 3, ...).

    Returns:
        The scaled chain ready for name generation.
    """
    chain: dict[str, dict] = {}

    for full_name in name_list:
        parts = full_name.split()
        _incr(chain, "parts", len(parts))

        for name in parts:
            _incr(chain, "name_len", len(name))

            # Record initial sequences of length 1 through min(order, len(name)).
            # For order=3 and "Achilles":
            #   initial_1 += "A", initial_2 += "Ac", initial_3 += "Ach"
            for o in range(1, min(order, len(name)) + 1):
                key = "initial" if o == 1 else f"initial_{o}"
                _incr(chain, key, name[:o])

            # Record transitions for EVERY order from 1 up to the configured max.
            # This ensures backoff always has data to fall back on.
            #
            # For each position k in the name, and each context length o:
            #   key = name[k-o : k]  (the o chars ending just before position k)
            #   token = name[k]      (the character at position k)
            for k in range(1, len(name)):
                for o in range(1, min(order, k) + 1):
                    context = name[k - o : k]
                    _incr(chain, context, name[k])

    return _scale(chain)


def _incr(chain: dict, key: str, token) -> None:
    """Increment chain[key][token] by 1, initializing if needed."""
    if key not in chain:
        chain[key] = {}
    if token not in chain[key]:
        chain[key][token] = 0
    chain[key][token] += 1


def _scale(chain: dict) -> dict:
    """Apply power scaling and pre-compute totals (identical to order-1)."""
    table_len = {}

    for key in chain:
        table_len[key] = 0
        for token in chain[key]:
            count = chain[key][token]
            weighted = math.floor(count ** 1.3)
            chain[key][token] = weighted
            table_len[key] += weighted

    chain["table_len"] = table_len
    return chain


# =============================================================================
# CORE ALGORITHM: Generating Names with Backoff
# =============================================================================

def markov_name(chain: dict, order: int) -> str:
    """
    Generate a name by walking the chain with backoff.

    The key difference from order-1: when picking the next character, we try
    the full N-character context first. If that context was never seen in
    training, we shorten it by one character and try again, down to 1 char.

    This is exactly the "stupid backoff" strategy used in traditional NLP
    n-gram language models (Brants et al., 2007). It never asks how much to
    trust the shorter context; it just takes it.

    Args:
        chain: A scaled Markov chain built by construct_chain().
        order: The maximum context length to try before backing off.

    Returns:
        A generated name string.
    """
    num_parts = _select(chain, "parts")
    name_parts = []

    for _ in range(num_parts):
        target_len = _select(chain, "name_len")

        # Pick initial sequence: try the longest initial first (backoff).
        # For order=3: try initial_3, then initial_2, then initial.
        name = None
        for o in range(order, 0, -1):
            key = "initial" if o == 1 else f"initial_{o}"
            val = _select(chain, key)
            if val is not None:
                name = val
                break

        if name is None:
            continue

        # Build the rest of the name, character by character with backoff.
        while len(name) < target_len:
            c = None

            # Try the longest context first, fall back to shorter.
            # For order=3 and name="Gand":
            #   Try "and" (order-3), then "nd" (order-2), then "d" (order-1)
            for o in range(min(order, len(name)), 0, -1):
                context = name[-o:]
                next_c = _select(chain, context)
                if next_c is not None:
                    c = next_c
                    break

            if c is None:
                break

            name += c

        name_parts.append(name)

    return " ".join(name_parts)


def _select(chain: dict, key: str):
    """Weighted random sampling (identical to order-1 version)."""
    total = chain["table_len"].get(key, 0)
    if not total:
        return None

    idx = random.randint(0, total - 1)

    acc = 0
    for token, weight in chain[key].items():
        acc += weight
        if acc > idx:
            return token

    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def get_chain(name_type: str, order: int) -> dict | None:
    """Retrieve or build+cache the chain for a given name type and order."""
    cache_key = f"{name_type}:{order}"
    if cache_key in _chain_cache:
        return _chain_cache[cache_key]

    name_list = NAME_SETS.get(name_type)
    if name_list:
        chain = construct_chain(name_list, order)
        _chain_cache[cache_key] = chain
        return chain

    return None


def generate_name(name_type: str, order: int) -> str:
    """Generate a single name of the given type and order."""
    chain = get_chain(name_type, order)
    if chain:
        return markov_name(chain, order)
    return ""


def generate_names(name_type: str, order: int, count: int) -> list[str]:
    """Generate multiple names of the given type and order."""
    return [generate_name(name_type, order) for _ in range(count)]


# =============================================================================
# DEMO: Compare orders 1, 2, and 3 side-by-side
# =============================================================================

def demo():
    print("=== Higher-Order Markov Chain Name Generator (Python) ===\n")
    print("Comparing order 1 vs 2 vs 3 on the same training data.")
    print("Higher orders capture more context but risk overfitting.\n")
    print("ORDER-1: 'Given the last 1 char, what comes next?'")
    print("ORDER-2: 'Given the last 2 chars, what comes next?'")
    print("ORDER-3: 'Given the last 3 chars, what comes next?'\n")
    print("Backoff: if order-N context is unseen, fall back to order-(N-1).\n")

    orders = [1, 2, 3]
    count = 10

    for name_type, names in NAME_SETS.items():
        print("=" * 60)
        print(f"  {name_type.upper()} ({len(names)} training names)")
        print("=" * 60)

        # Generate names for each order
        columns = {}
        for order in orders:
            columns[order] = generate_names(name_type, order, count)

        # Print header
        print(f"  {'Order 1':<20}{'Order 2':<20}{'Order 3':<20}")
        print(f"  {'-' * 58}")

        # Print side-by-side
        for i in range(count):
            col1 = (columns[1][i] or "")[:19].ljust(20)
            col2 = (columns[2][i] or "")[:19].ljust(20)
            col3 = (columns[3][i] or "")[:19].ljust(20)
            print(f"  {col1}{col2}{col3}")

        print()

    # Show a peek inside an order-2 chain to illustrate multi-char keys
    print("--- PEEK INSIDE THE ORDER-2 CHAIN (greek) ---")
    chain = get_chain("greek", 2)

    # Show a 2-char key
    if chain and "ch" in chain:
        print('After "ch", the possible next characters:')
        total = chain["table_len"]["ch"]
        for char, weight in chain["ch"].items():
            pct = (weight / total) * 100
            print(f"  '{char}' -> weight {weight} ({pct:.1f}%)")

    # Show a 1-char key for comparison
    if chain and "h" in chain:
        print('\nCompare: after just "h" (order-1 context):')
        total = chain["table_len"]["h"]
        for char, weight in chain["h"].items():
            pct = (weight / total) * 100
            print(f"  '{char}' -> weight {weight} ({pct:.1f}%)")
        print(
            '\nNotice: "th" has a much more focused distribution than just "h"!'
        )
        print(
            "This is the power of higher-order context — more specific predictions.\n"
        )

    print("--- OBSERVATIONS ---")
    print("Order 1: Very random, often gibberish. Each char is independent.")
    print("Order 2: Noticeably better. Captures common pairs like 'th', 'ar'.")
    print("Order 3: Even more realistic, but may copy training names.")
    print("         (This is OVERFITTING: too much memorization, too little")
    print("         generalization. Raising the order trades novelty for fluency.)\n")


if __name__ == "__main__":
    demo()
