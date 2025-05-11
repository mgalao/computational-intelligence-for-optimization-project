"""
This module contains various crossover functions for combining the genetic material
of two parent individuals in a genetic algorithm. The crossovers include:
1. pmx_crossover: Selects a segment from each parent and swaps it
                  to produce two offspring, resolving duplicate entries.
2. fitness_based_slot_crossover: Selects the best-performing slots from both parents
   based on local slot fitness, builds an offspring from the top-performing slots,
   and then repairs the individual to ensure each artist appears exactly once.
"""

from evolution.entities import Individual
from data.import_data import artists
from copy import deepcopy
import random
from random import choice
from collections import Counter

def pmx_crossover(
    p1: 'Individual',
    p2: 'Individual',
) -> list[list[int]]:

    """
    Partially Mapped Crossover (PMX) implementation with three segments.

    This implementation splits the chromosome into three segments.
    For each child, segments 1 and 3 are copied from one parent, while segment 2
    (the crossover window) is copied from the other. Duplicates are resolved using
    position mapping to ensure each artist ID appears exactly once.

    Args:
        p1: First parent for crossover (Individual).
        p2: Second parent for crossover (Individual).

    Returns:
        offspring1: First offspring resulting from the crossover (Individual).
        offspring2: Second offspring resulting from the crossover (Individual).
    """
    
    # Get the representation of the parents
    p1_repr = p1.repr
    p2_repr = p2.repr

    # Flatten 2D representations
    p1_flat = [artist for row in p1_repr for artist in row]
    p2_flat = [artist for row in p2_repr for artist in row]
    
    size = len(p1_flat)
    
    # Print parent info
    print("Parent 1:", p1_flat)
    print("Parent 2:", p2_flat)

    # Crossover points
    cx1, cx2 = sorted(random.sample(range(size), 2))
    print(f"Crossover points: cx1={cx1}, cx2={cx2}")

    # Initial children
    child1 = [None] * size
    child2 = [None] * size

    # Middle segment
    child1[cx1:cx2] = p2_flat[cx1:cx2]
    child2[cx1:cx2] = p1_flat[cx1:cx2]

    print(f"Child1 after crossover segment: {child1}")
    print(f"Child2 after crossover segment: {child2}")

    # Mappings
    mapping1 = {p2_flat[i]: p1_flat[i] for i in range(cx1, cx2)}
    mapping2 = {p1_flat[i]: p2_flat[i] for i in range(cx1, cx2)}

    print("Mapping1 (for child1):", mapping1)
    print("Mapping2 (for child2):", mapping2)

    segment1 = set(child1[cx1:cx2])
    segment2 = set(child2[cx1:cx2])

    def resolve_mapping(value, mapping, segment, filled):
        if value in segment:
            resolved_value = mapping.get(value, value)
            while resolved_value in filled:
                resolved_value = mapping.get(resolved_value, resolved_value)
            return resolved_value
        return value

    filled_values1 = set(child1)
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        val = p1_flat[i]
        child1[i] = resolve_mapping(val, mapping1, segment1, filled_values1)
        filled_values1.add(child1[i])

    filled_values2 = set(child2)
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        val = p2_flat[i]
        child2[i] = resolve_mapping(val, mapping2, segment2, filled_values2)
        filled_values2.add(child2[i])

    # Reshape to original dimensions
    num_rows = len(p1_repr)
    num_cols = len(p2_repr[0])
    child1_matrix = [child1[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]
    child2_matrix = [child2[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

    return child1_matrix, child2_matrix

def fitness_based_slot_crossover(
    p1: 'Individual',
    p2: 'Individual',
) -> list[list[int]]:
    """
    Fitness Based Slot Crossover implementation.

    This crossover selects the best-performing slots from both parents
    based on local slot fitness, builds an offspring from the top-performing slots,
    and then repairs the individual to ensure each artist appears exactly once.

    The repair phase replaces duplicates with missing artists, starting from the 
    weakest slots to preserve strong ones.

    Args:
        p1: First parent.
        p2: Second parent.
        verbose: If True, prints detailed steps.

    Returns:
        offspring: Valid child with one artist per slot.
        None: Placeholder for a second child (not used).
    """

    # Get the representation of the parents
    p1_repr = p1.repr
    p2_repr = p2.repr

    num_slots = len(p1_repr)
    num_stages = len(p1_repr[0])

    # Score and collect all slots
    scored_slots = []

    for slot_idx in range(num_slots):
        slot_p1 = deepcopy(p1_repr[slot_idx])
        slot_p2 = deepcopy(p2_repr[slot_idx])
        score_p1 = p1.get_slot_fitness(slot_idx)
        score_p2 = p2.get_slot_fitness(slot_idx)
        scored_slots.append((score_p1, slot_p1))
        scored_slots.append((score_p2, slot_p2))

    # Select top slots by fitness
    scored_slots.sort(key=lambda x: x[0], reverse=True)
    selected_slots = scored_slots[:num_slots]

    # Print selected slots
    print("\n--- Top Slot Fitness Scores ---")
    for i, (score, slot) in enumerate(selected_slots):
        print(f"Slot {i+1:2d} | Fitness: {score:.4f} | Artists: {slot}")
    print("-----------------------------------\n")

    # Sort weakest to strongest to repair weakest first
    selected_slots.sort(key=lambda x: x[0])  
    offspring_repr = deepcopy([slot for _, slot in selected_slots])

    # Repair: ensure valid permutation 
    # Flatten to detect duplicates
    flat = [artist for row in offspring_repr for artist in row]
    counts = Counter(flat)

    all_artists = set(artists.index)
    present_artists = set(flat)
    missing = list(all_artists - present_artists)

    # Print counts
    print(f"Total artists expected: {len(all_artists)}")
    print(f"Duplicates detected: {[k for k, v in counts.items() if v > 1]}")
    print(f"Missing artists: {missing}\n")

    used_missing = set()
    used_duplicates = set()

    for i in range(num_slots):
        for j in range(num_stages):
            artist = offspring_repr[i][j]

            # Replace if it's a duplicate (not yet fixed)
            if counts[artist] > 1 and artist not in used_duplicates:
                # Choose a missing artist that hasn't been used
                available_missing = list(set(missing) - used_missing)
                if not available_missing:
                    break  # No more missing artists to assign

                replaced_with = choice(available_missing)
                # Print replacement details
                print(f"Replacing duplicate artist {artist} in slot {i}, stage {j} with missing artist {replaced_with}")

                offspring_repr[i][j] = replaced_with
                counts[artist] -= 1
                used_duplicates.add(artist)
                used_missing.add(replaced_with)

    # Print final offspring representation
    print("\n--- Final Offspring Representation ---")
    for idx, row in enumerate(offspring_repr):
        print(f"Slot {idx+1}: {row}")
    print("------------------------------------------\n")

    return offspring_repr, None