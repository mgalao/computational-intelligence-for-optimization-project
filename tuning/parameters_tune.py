""""
This module implements a random search algorithm for hyperparameter tuning of a genetic algorithm.
It samples hyperparameters from specified ranges and evaluates the performance of the genetic algorithm
using those hyperparameters.
It returns a DataFrame containing the results of the tuning process.
"""

from algorithms.genetic_algorithm.entities import Population
from algorithms.genetic_algorithm.selection import (tournament_selection)
from algorithms.genetic_algorithm.crossover import (pmx_crossover)
from algorithms.genetic_algorithm.mutation import (n_swap_mutation)
from algorithms.genetic_algorithm.genetic_algorithm import genetic_algorithm

from utils import *

# Fixed Parameters
selection_base = tournament_selection
crossover = pmx_crossover
mutation_base = n_swap_mutation
n_runs = 30
n_generations = 100
pop_size = 50
elitism = True
maximization = True

n_iterations = 25


def random_search(
    mutation_base=mutation_base,
    crossover=crossover,
    selection_base=selection_base,
    n_iterations=n_iterations,
    n_runs=n_runs,
    n_generations=n_generations,
    pop_size=pop_size,
    elitism=elitism,
    maximization=maximization,
    xo_range=(0.6, 1.0),             
    mut_range=(0.05, 0.4),           
    swap_range=(2, 5),               
    tournament_range=(2, 10),       
    verbose_ga=True,
    mode='summary',
    seed=None
):
    """
    Perform a random search for hyperparameter tuning of a genetic algorithm.
    Args:
        mutation_base (function): The base mutation function to be used.
        crossover (function): The crossover function to be used.
        selection_base (function): The base selection function to be used.
        n_iterations (int): The number of iterations for the random search.
        n_runs (int): The number of runs for each configuration.
        n_generations (int): The number of generations for the genetic algorithm.
        pop_size (int): The population size for the genetic algorithm.
        elitism (bool): Whether to use elitism in the genetic algorithm.
        maximization (bool): Whether to maximize or minimize the fitness function.
        xo_range (tuple): The range for the crossover probability.
        mut_range (tuple): The range for the mutation probability.
        swap_range (tuple): The range for the number of swaps in mutation.
        tournament_range (tuple): The range for the tournament size in selection.
        verbose_ga (bool): If True, prints detailed information during the genetic algorithm execution.
        mode (str): The mode of operation ('detailed' or 'summary').
        seed (int, optional): Random seed for reproducibility.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the random search.
    """

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    tried_configs = set()
    results = []

    while len(results) < n_iterations:
        # Generate random hyperparameters
        xo_prob = round(random.uniform(*xo_range), 2)
        mut_prob = round(random.uniform(*mut_range), 2)
        n_swaps = random.randint(*swap_range)
        tournament_size = random.randint(*tournament_range)

        # Check for duplicates
        config_key = (xo_prob, mut_prob, n_swaps)
        if config_key in tried_configs:
            continue
        tried_configs.add(config_key)

        print(f'Configuration {len(results)+1}: xo_prob={xo_prob}, mut_prob={mut_prob}, n_swaps={n_swaps}, tournament_size={tournament_size}')
        run_histories = []

        #  Run n_runs for each configuration
        for run in range(n_runs):
            print(f'  Run {run + 1}/{n_runs}')

            # Apply partial to mutation operator
            mutation = partial(mutation_base, n_swaps=n_swaps)
            selection = partial(selection_base, tournament_size=tournament_size)

            # Run GA
            population = Population(population_size=pop_size, crossover_function=crossover, mutation_function=mutation)

            # Run the genetic algorithm with the specified parameters
            fitness_history, _ = genetic_algorithm(
                initial_population=population,
                max_gen=n_generations,
                selection_algorithm=selection,
                maximization=maximization,
                xo_prob=xo_prob,
                mut_prob=mut_prob,
                elitism=elitism,
                verbose_ga=verbose_ga
            )

            # Append the fitness history to the run histories
            run_histories.append(fitness_history)

        #  Store Results 
        if mode == 'detailed':
            results.append({
                'xo_prob': xo_prob,
                'mut_prob': mut_prob,
                'n_swaps': n_swaps,
                'tournament_size': tournament_size,
                'fitness_history': json.dumps(run_histories)
            })
        else:
            best_fitnesses = [max(h) if maximization else min(h) for h in run_histories]
            avg_best = sum(best_fitnesses) / n_runs
            results.append({
                'xo_prob': xo_prob,
                'mut_prob': mut_prob,
                'n_swaps': n_swaps,
                'tournament_size': tournament_size,
                'avg_best_fitness': avg_best
            })

    return pd.DataFrame(results)
