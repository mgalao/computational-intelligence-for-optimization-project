from evolution.entities import Individual
from data.import_data import artists
import random
from collections import Counter
from copy import deepcopy

def pmx_crossover(p1, p2):
    """
    Partially Mapped Crossover (PMX) implementation with three segments.

    This implementation splits the chromosome into three segments.
    For each child, segments 1 and 3 are copied from one parent, while segment 2
    (the crossover window) is copied from the other. Duplicates are resolved using
    position mapping to ensure each artist ID appears exactly once.

    Args:
        p1: First parent for crossover.
        p2: Second parent for crossover.

    Returns:
        offspring1: First offspring resulting from the crossover.
        offspring2: Second offspring resulting from the crossover.
    """
    # Flatten parents
    p1_flat = [artist for row in p1.repr for artist in row]
    p2_flat = [artist for row in p2.repr for artist in row]
    size = len(p1_flat)

    # Random crossover points
    cx1, cx2 = sorted(random.sample(range(size), 2))

    # Create childs
    child1 = [None] * size
    child2 = [None] * size

    # Copy crossover segment from other parent in the middle
    child1[cx1:cx2] = p2_flat[cx1:cx2]
    child2[cx1:cx2] = p1_flat[cx1:cx2]

    # Build mapping from the crossover segment
    mapping1 = {p2_flat[i]: p1_flat[i] for i in range(cx1, cx2)}
    mapping2 = {p1_flat[i]: p2_flat[i] for i in range(cx1, cx2)}

    # Convert middle segments to sets
    segment1 = set(child1[cx1:cx2])
    segment2 = set(child2[cx1:cx2])

    # Functions to resolve the conflicts
    def resolve_mapping(value, mapping, segment):
        if value in segment:
            return mapping.get(value, value)
        return value

    # Fill child1 with values from p1, resolving conflicts
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        val = p1_flat[i]
        child1[i] = resolve_mapping(val, mapping1, segment1)

    # Fill child2 with values from p2, resolving conflicts
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        val = p2_flat[i]
        child2[i] = resolve_mapping(val, mapping2, segment2)

    # Reshape to matrix
    num_rows = len(p1.repr)
    num_cols = len(p1.repr[0])
    child1_matrix = [child1[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]
    child2_matrix = [child2[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

    return Individual(repr=child1_matrix), Individual(repr=child2_matrix)


def eager_breeder_crossover(p1, p2, verbose=True):
    """
    Eager Breeder Crossover (EBC) implementation.

    This crossover selects the best-performing slots (rows) from both parents
    based on local slot fitness, builds an offspring from the top-performing slots,
    and then repairs the individual to ensure each artist appears exactly once.

    The repair phase replaces duplicates (artists appearing more than once) with 
    missing artists, starting from the weakest slots to preserve strong ones.

    Args:
        p1 (Individual): First parent.
        p2 (Individual): Second parent.
        verbose (bool): If True, prints detailed steps.

    Returns:
        offspring (Individual): Valid child with one artist per slot.
        None: Placeholder for a second child (not used).
    """

    num_slots = len(p1.repr)
    num_stages = len(p1.repr[0])

    # Score and collect all slots
    scored_slots = []

    for slot_idx in range(num_slots):
        slot_p1 = deepcopy(p1.repr[slot_idx])
        slot_p2 = deepcopy(p2.repr[slot_idx])
        score_p1 = p1.get_slot_fitness(slot_idx)
        score_p2 = p2.get_slot_fitness(slot_idx)
        scored_slots.append((score_p1, slot_p1))
        scored_slots.append((score_p2, slot_p2))

    # Select top slots by fitness
    scored_slots.sort(key=lambda x: x[0], reverse=True)
    selected_slots = scored_slots[:num_slots]

    if verbose:
        print("\n--- Top Slot Fitness Scores (Before Repair) ---")
        for i, (score, slot) in enumerate(selected_slots):
            print(f"Slot {i+1:2d} | Fitness: {score:.4f} | Artists: {slot}")
        print("------------------------------------------------\n")

    # Sort weakest-to-strongest to repair weakest first
    selected_slots.sort(key=lambda x: x[0])  # Ascending order
    offspring_repr = deepcopy([slot for _, slot in selected_slots])

    # Repair: ensure valid permutation 
    # Flatten to detect duplicates
    flat = [artist for row in offspring_repr for artist in row]
    counts = Counter(flat)

    all_artists = set(artists.index)
    present_artists = set(flat)
    missing = list(all_artists - present_artists)

    used = set()
    missing_idx = 0

    if verbose:
        print(f"Total artists expected: {len(all_artists)}")
        print(f"Duplicates detected: {[k for k, v in counts.items() if v > 1]}")
        print(f"Missing artists: {missing}\n")

    for i in range(num_slots):
        for j in range(num_stages):
            artist = offspring_repr[i][j]

            # Replace if it's a duplicate (not yet fixed)
            if counts[artist] > 1 and artist not in used:
                replaced_with = missing[missing_idx]
                if verbose:
                    print(f"Replacing duplicate artist {artist} in slot {i}, stage {j} with missing artist {replaced_with}")
                offspring_repr[i][j] = replaced_with
                counts[artist] -= 1
                missing_idx += 1
                used.add(artist)

                # Stop early if no more missing artists
                if missing_idx >= len(missing):
                    break
        if missing_idx >= len(missing):
            break

    if verbose:
        print("\n--- Final Offspring Representation ---")
        for idx, row in enumerate(offspring_repr):
            print(f"Slot {idx+1}: {row}")
        print("------------------------------------------------\n")

    return Individual(repr=offspring_repr), None