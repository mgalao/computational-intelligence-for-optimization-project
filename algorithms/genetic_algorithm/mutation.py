"""
This module contains various mutation functions for modifying the representation of individuals
in a genetic algorithm. The mutations include:
1. n_swap_mutation: Swaps elements between different rows in the matrix representation.
2. scramble_mutation: Scrambles a random-length segment within a randomly selected column (stage).
3. prime_slot_swap_mutation: Swaps the prime slot (last row) with another randomly selected slot (row).
4. preserve_best_slots_mutation: Preserves high-fitness time slots (rows) based on their individual slot fitness scores, 
   and randomly redistributes the artists from the remaining slots.
"""

from algorithms.genetic_algorithm.entities import Individual
from utils import *

def n_swap_mutation(
    individual: 'Individual',
    mut_prob: float,
    n_swaps: int = 2,
    verbose: bool = False
) -> list[list[int]]:
    """
    Applies n swap mutations to the given 2D matrix representation with a specified probability.
    The mutation involves swapping elements between different rows (i.e., different time slots).

    Args:
        individual (Individual): The individual solution.
        mut_prob (float): Probability of mutation.
        n_swaps (int): Number of swaps to perform.

    Returns:
        list of lists: Mutated representation.
    """

    # Copy the original representation to avoid modifying it
    new_representation = deepcopy(individual.repr)

    # Store the number of rows and columns
    num_rows = len(new_representation)
    num_cols = len(new_representation[0])
    total_elements = num_rows * num_cols

    # n_swaps must be a positive integer and cannot exceed half the number of elements
    if not isinstance(n_swaps, int) or n_swaps < 1:
        raise ValueError("n_swaps must be a positive integer.")
    if n_swaps > total_elements // 2:
        raise ValueError(f"n_swaps cannot exceed {total_elements // 2} for a matrix of size {num_rows}x{num_cols}.")

    # Perform mutation with the given probability
    if random.random() <= mut_prob:
        for i in range(n_swaps):
            while True:
                # Choose two different element positions in the matrix
                # divmod(x, y) returns (quotient -> x // y, reminder -> x % y)
                r1, c1 = divmod(random.randint(0, total_elements - 1), num_cols)
                r2, c2 = divmod(random.randint(0, total_elements - 1), num_cols)
                
                # Ensure the swap occurs between elements of different rows (i.e., different time slots)
                if r1 != r2:
                    break
            
            # Store the values to be swapped
            val1 = new_representation[r1][c1]
            val2 = new_representation[r2][c2]

            # Print the swap details
            if verbose:
                print(f"Swap {i+1}: ({r1},{c1}) [{val1}] <--> ({r2},{c2}) [{val2}]")

            # Perform the swap
            new_representation[r1][c1], new_representation[r2][c2] = val2, val1

    return new_representation

def scramble_mutation(
    individual: 'Individual',
    mut_prob: float,
    max_segment_length: int = 4,
    verbose: bool = False
) -> list[list[int]]:
    """
    Applies scramble mutation by shuffling a random-length segment
    within a randomly selected column (stage).

    Args:
        individual (Individual): The individual solution.
        mut_prob (float): Probability of mutation.
        segment_length (int): Length of the segment to scramble.

    Returns:
        list of lists: Mutated matrix representation.
    """

    # Copy the original representation to avoid modifying it
    new_representation = deepcopy(individual.repr)

    # Store the number of rows and columns
    num_rows = len(new_representation)
    num_cols = len(new_representation[0])

    # max_segment_length must be a positive integer (greater than or equal to 2) and cannot exceed the number of rows
    if not isinstance(max_segment_length, int) or max_segment_length < 2:
        raise ValueError("max_segment_length must be an integer greater than or equal to 2.")
    if max_segment_length > num_rows:
        raise ValueError(f"max_segment_length cannot exceed number of rows ({num_rows}).")

    # Perform mutation with the given probability
    if random.random() <= mut_prob:
        # Select a random column (stage), random segment length and random start index
        col = random.randint(0, num_cols - 1)
        actual_length = random.randint(2, max_segment_length)
        start = random.randint(0, num_rows - actual_length)
        end = start + actual_length

        # Extract the segment to be scrambled
        segment = [new_representation[r][col] for r in range(start, end)]

        # Print the details of the mutation
        if verbose:
            print(f"Stage: {col}, Slots: {start}-{end-1}, Length: {actual_length}")
            print(f"Before: {segment}")

        random.shuffle(segment)

        if verbose:
            print(f"After: {segment}")

        # Place the scrambled segment back into the representation
        for i, r in enumerate(range(start, end)):
            new_representation[r][col] = segment[i]

    return new_representation

def prime_slot_swap_mutation(
    individual: 'Individual',
    mut_prob: float,
    verbose: bool = False
) -> list[list[int]]:
    """
    Swaps the prime slot (last row) with another randomly selected slot (row).

    Args:
        individual (Individual): The individual solution.
        mut_prob (float): Probability of mutation.

    Returns:
        list of lists: Mutated matrix representation.
    """

    # Copy the original representation to avoid modifying it
    new_representation = deepcopy(individual.repr)

    # Store the number of rows
    num_rows = len(new_representation)

    # Perform mutation with the given probability
    if random.random() <= mut_prob:
        # Select a random index for the prime slot (last row)
        prime_idx = num_rows - 1
        other_idx = random.randint(0, prime_idx - 1)

        # Print the details before the swap
        if verbose:
            print(f"Swapping row {prime_idx} (prime slot) with row {other_idx}")
            print(f"Before - Prime Slot: {new_representation[prime_idx]}")
            print(f"Before - Other Slot: {new_representation[other_idx]}")

        # Swap the prime slot with the randomly selected slot
        new_representation[prime_idx], new_representation[other_idx] = (
            new_representation[other_idx],
            new_representation[prime_idx],
        )

        # Print the details after the swap
        if verbose:
            print(f"After - Prime Slot: {new_representation[prime_idx]}")
            print(f"After - Other Slot : {new_representation[other_idx]}")

    return new_representation

def preserve_best_slots_mutation(
    individual: 'Individual',
    mut_prob: float,
    keep_ratio: float = 0.5,
    verbose: bool = False
) -> list[list[int]]:
    """
    Preserves high-fitness time slots (rows) based on their individual slot fitness scores,
    and randomly redistributes the artists from the remaining slots.

    Args:
        individual (Individual): The individual solution.
        mut_prob (float): Probability of mutation.
        keep_ratio (float): Fraction of the best-performing slots (rows) to preserve (value between 0 and 1).

    Returns:
        Individual: The mutated individual with updated representation.
    """

    # Copy the original representation to avoid modifying it
    new_representation = deepcopy(individual.repr)

    # Store the number of rows and columns
    num_rows = len(new_representation)
    num_cols = len(new_representation[0])

    # Perform mutation with the given probability
    if random.random() <= mut_prob:
        # Evaluate fitness for each slot
        slot_fitnesses = [(i, individual.get_slot_fitness(i)) for i in range(num_rows)]

        # Sort and decide which rows (slots) to keep
        slot_fitnesses.sort(key=lambda x: x[1], reverse=True)
        num_to_keep = int(keep_ratio * num_rows)
        rows_to_keep = {i for i, _ in slot_fitnesses[:num_to_keep]}

        # Print the fitness scores and the rows to keep
        if verbose:
            print(f"Preserving rows: {sorted(rows_to_keep)}")

        # Gather remaining artists and shuffle them
        remaining_artists = [
            new_representation[i][j]
            for i in range(num_rows)
            for j in range(num_cols)
            if i not in rows_to_keep
        ]
        random.shuffle(remaining_artists)

        # Replace the non-kept rows with shuffled artists
        idx = 0
        for i in range(num_rows):
            if i not in rows_to_keep:
                for j in range(num_cols):
                    new_representation[i][j] = remaining_artists[idx]
                    idx += 1

    return new_representation