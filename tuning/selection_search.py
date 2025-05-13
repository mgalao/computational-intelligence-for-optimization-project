from evolution.entities import Population
from evolution.selection import (fitness_proportionate_selection, ranking_selection,tournament_selection )
from evolution.crossover import ( pmx_crossover, fitness_based_slot_crossover )
from evolution.mutation import (n_swap_mutation,scramble_mutation,prime_slot_swap_mutation,preserve_best_slots_mutation)
from evolution.algorithm import genetic_algorithm

import pandas as pd
from itertools import product
from random import sample
from evolution.algorithm import genetic_algorithm
import numpy as np

crossover_operators = [pmx_crossover, fitness_based_slot_crossover]
mutation_operators = [n_swap_mutation, scramble_mutation, prime_slot_swap_mutation, preserve_best_slots_mutation]

# All combinations of crossover and mutation operators
operator_combinations= list(product(crossover_operators, mutation_operators))

# Add the selection methods to the combinations
fitness_proportionate_selection_combinations = list(product([fitness_proportionate_selection], operator_combinations))
ranking_selection_combinations = list(product([ranking_selection], operator_combinations))
tournament_selection_combinations = list(product([tournament_selection], operator_combinations))

# Combinations of selection, crossover and mutation to be tested
selection_strategies = [fitness_proportionate_selection, ranking_selection, tournament_selection]

# ---------------------- Fixed Hyperparameters ---------------------- #
# Genetic operators and elitism
xo_prob = 0.9
mut_prob = 0.1

# GA evolution
nr_runs = 30
n_generations = 100
pop_size = 50

# Problem specific

results = pd.DataFrame()


for (selection, (crossover, mutation)) in selection_strategies:

    # Save the name of the combination
    combination_name = f'{selection.__name__}|{crossover.__name__}|{mutation.__name__}'

    # Save results of each run
    comb_results = []

    for run_nr in range(nr_runs):
        print(f'----------- Run_{run_nr + 1} of comb {combination_name}')

        # Create a population
        initial_population = Population(pop_size = pop_size, crossover_function=crossover, mutation_function=mutation)

        # Run the GA
        best_fitness = genetic_algorithm(initial_population=initial_population,
                            # max_gen=100,
                            max_gen=1,
                            selection_algorithm=fitness_proportionate_selection,
                            maximization=True,
                            xo_prob=0.4,
                            mut_prob=0.7,
                            elitism=True,
                            verbose=True)

        # Save the best fitnesses of each generation
        comb_results.append(best_fitness)

    results[combination_name] = np.median(np.transpose(comb_results), axis = 1)


results.to_csv('selection_results.csv')

