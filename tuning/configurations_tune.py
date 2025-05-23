""""
This module implements a grid search for tuning the parameters of a genetic algorithm.
It evaluates different combinations of crossover, mutation, and selection methods to find
the best configuration based on the fitness of the solutions.
The grid search runs multiple iterations for each configuration and collects the results for analysis.
"""

from algorithms.genetic_algorithm.entities import Population
from algorithms.genetic_algorithm.selection import (fitness_proportionate_selection, ranking_selection, tournament_selection)
from algorithms.genetic_algorithm.crossover import (pmx_crossover, fitness_based_slot_crossover)
from algorithms.genetic_algorithm.mutation import (n_swap_mutation,scramble_mutation,prime_slot_swap_mutation,preserve_best_slots_mutation)
from algorithms.genetic_algorithm.genetic_algorithm import genetic_algorithm
from algorithms.genetic_algorithm.genetic_algorithm import genetic_algorithm
from utils import *

# Define the crossover, mutation, and selection operators
crossover_operators = [pmx_crossover, fitness_based_slot_crossover]
mutation_operators = [n_swap_mutation, scramble_mutation, prime_slot_swap_mutation, preserve_best_slots_mutation]
selection_methods = [fitness_proportionate_selection, ranking_selection, tournament_selection]

# All combinations of crossover and mutation operators
configurations= list(product(crossover_operators, mutation_operators, selection_methods))

# Fixed Parameters 
xo_prob = 0.9
mut_prob = 0.1
n_generations = 100
pop_size = 50
elitism = True
maximization = True

def grid_search(configurations, mode='detailed', n_runs=30, verbose_ga=True):
    """
    Perform a grid search over the specified configurations of crossover, mutation, and selection methods.
    Args:
        configurations (list): A list of tuples containing the crossover, mutation, and selection methods.
        mode (str): The mode of operation ('detailed' or 'summary').
        n_runs (int): The number of runs for each configuration.
        verbose_ga (bool): If True, prints detailed information during the genetic algorithm execution.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the grid search.
    """

    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame()

    # Iterate over each configuration
    for (crossover, mutation, selection) in configurations:
        # Create a unique name for the configuration
        configuration_name = f'Selection: {selection.__name__} | Crossover: {crossover.__name__} | Mutation: {mutation.__name__}'

        configuration_results = []
        count=0

        # Run the genetic algorithm for the specified number of runs
        while (count<n_runs):
            print(f'Run {count + 1 } - {configuration_name}')

            # Create a new population with the specified crossover and mutation functions
            initial_population = Population(population_size = pop_size, crossover_function=crossover, mutation_function=mutation)
            
            # Run the genetic algorithm with the specified parameters
            fitness_history, _ = genetic_algorithm(initial_population=initial_population,
                                max_gen=n_generations,
                                selection_algorithm=selection,
                                maximization=maximization,
                                xo_prob=xo_prob,
                                mut_prob=mut_prob,
                                elitism=elitism,
                                verbose_ga=verbose_ga)
            
            # Append the fitness history to the configuration results
            configuration_results.append(fitness_history)
            count += 1

        # Store the results in the DataFrame
        if mode =='detailed':
            results[configuration_name] = [json.dumps(gen) for gen in configuration_results]
        elif mode == 'summary':
            best_fitnesses = [max(run) for run in configuration_results]
            results[configuration_name] = best_fitnesses

    return results