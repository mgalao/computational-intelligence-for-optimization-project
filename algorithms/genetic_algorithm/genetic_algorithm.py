"""
This module implements a genetic algorithm for optimizing a population of solutions.
It includes functions for selection, crossover, mutation, and elitism.
"""

from algorithms.genetic_algorithm.entities import *
from utils import *

def genetic_algorithm(
    initial_population: 'Population',
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = True,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose_ga: bool = True,
    adapt_on_stable: bool = False,
    params: Optional[dict] = None,
    stability_window: int = 5,
    stability_epsilon: float = 1e-3
) -> 'Individual':
    """
    Executes a genetic algorithm to optimize a population of solutions, with optional dynamic adaptation 
    of mutation and selection parameters when the population's fitness stabilizes.

    Args:
        initial_population (Population): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals. Must support optional 'params' if adaptation is enabled.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes it. Defaults to True.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose_ga (bool, optional): If True, prints detailed logs for each generation. Defaults to True.
        adapt_on_stable (bool, optional): If True, enables dynamic adaptation of mutation and selection parameters when fitness stabilizes. Defaults to False.
        params (dict, optional): Dictionary with keys 'n_swaps', 'max_swaps', 'tournament_size', and 'min_tournament', used when adaptation is enabled.
        stability_window (int, optional): Number of generations over which fitness stabilization is evaluated. Defaults to 5.
        stability_epsilon (float, optional): Threshold for standard deviation of recent fitness values to trigger adaptation. Defaults to 1e-3.

    Returns:
        Tuple[List[float], Individual]: A tuple containing the list of best fitness values across generations 
        and the best individual found in the final generation.
    """

    # Initialize a population with N individuals
    population = initial_population
    fitness_history = []
    last_adaptation_gen = 0
    adaptation_cooldown = 5

     # Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose_ga:
            print(f'\n-------------- Generation: {gen} --------------')

        # Create an empty population P'
        new_population = []

        # If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(deepcopy(get_best_ind(population, maximization)))

        # Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population, params=params) if adapt_on_stable else selection_algorithm(population)
            second_ind = selection_algorithm(population, params=params) if adapt_on_stable else selection_algorithm(population)

            if verbose_ga:
                print(f'\nSelected individuals:\n{first_ind}\n{second_ind}\n')
            
            # Choose an operator between crossover and replication
            # Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
                if verbose_ga:
                    print(f'Applied crossover - offspring:')
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
                if verbose_ga:
                    print(f'Applied replication - offspring:')

            if verbose_ga:
                print(f'{offspring1}\n{offspring2}')

            # Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob, params=params) if adapt_on_stable else offspring1.mutation(mut_prob)
            
            # Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose_ga:
                print(f'\nFirst mutated individual:\n{first_new_ind}')

            if offspring2 is not None and len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob, params=params) if adapt_on_stable else offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose_ga:
                    print(f'Second mutated individual:\n{second_new_ind}')

        # Replace P with P'
        population = new_population
        best_fitness = get_best_ind(population, maximization).fitness()
        fitness_history.append(best_fitness)

        if verbose_ga:
            print(f'\nBest individual fitness: {best_fitness:.4f}\n')

        # EXTRA: Adaptation
        if adapt_on_stable and gen > stability_window:
            recent = fitness_history[-stability_window:]
            if np.std(recent) < stability_epsilon and (gen - last_adaptation_gen) >= adaptation_cooldown:
                if verbose_ga:
                    print("Fitness plateau detected — adapting parameters...")

                # Adjust mutation swaps
                params["n_swaps"] = min(params["n_swaps"] + 1, params.get("max_swaps", 10))

                # Reduce tournament size (lower selection pressure)
                params["tournament_size"] = max(params["tournament_size"] - 1, params.get("min_tournament", 2))

                last_adaptation_gen = gen

    return fitness_history, get_best_ind(population, maximization)
