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
import json
import csv


crossover_operators = [pmx_crossover, fitness_based_slot_crossover]
mutation_operators = [n_swap_mutation, scramble_mutation, prime_slot_swap_mutation, preserve_best_slots_mutation]
selection_methods = [fitness_proportionate_selection, ranking_selection,tournament_selection]

# All combinations of crossover and mutation operators
configurations= list(product(crossover_operators, mutation_operators, selection_methods))

# Fixed Parameters 
# Genetic operators and elitism
xo_prob = 0.9
mut_prob = 0.1

# GA evolution
n_generations = 100
pop_size = 50
elitism = True
verbose = True
maximization = True

def grid_search(configurations, mode='detailed', n_runs=5):

    results = pd.DataFrame()

    for (crossover, mutation, selection) in configurations:
        configuration_name = f'Selection: {selection.__name__} | Crossover: {crossover.__name__} | Mutation: {mutation.__name__}'

        configuration_results = []

        count=0

        while (count<n_runs):
            print(f'Run {count + 1 } - {configuration_name}')

            initial_population = Population(population_size = pop_size, crossover_function=crossover, mutation_function=mutation)
            
            fitness_history, _ = genetic_algorithm(initial_population=initial_population,
                                max_gen=n_generations,
                                selection_algorithm=selection,
                                maximization=maximization,
                                xo_prob=xo_prob,
                                mut_prob=mut_prob,
                                elitism=elitism,
                                verbose=verbose)
            
            configuration_results.append(fitness_history)
            count += 1

        if mode =='detailed':
            results[configuration_name] = [json.dumps(gen) for gen in configuration_results]
        elif mode == 'summary':
            results[configuration_name] = np.median(configuration_results, axis=1)

    filename='results_detailed.csv' if mode == 'detailed' else 'results_summary.csv'
    results.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC if mode == 'detailed' else csv.QUOTE_MINIMAL)

