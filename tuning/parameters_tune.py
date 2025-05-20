from algorithms.genetic_algorithm.entities import Population
from algorithms.genetic_algorithm.selection import (fitness_proportionate_selection, ranking_selection, tournament_selection)
from algorithms.genetic_algorithm.crossover import (pmx_crossover, fitness_based_slot_crossover)
from algorithms.genetic_algorithm.mutation import (n_swap_mutation,scramble_mutation,prime_slot_swap_mutation,preserve_best_slots_mutation)
from algorithms.genetic_algorithm.genetic_algorithm import genetic_algorithm

from utils import *

# --------------------- Fixed Hyperparameters -------------------- #
selection_base = tournament_selection
crossover = pmx_crossover
mutation_base = n_swap_mutation

n_runs = 30
n_generations = 100
pop_size = 50
elitism = True
maximization = True

n_iterations = 2

def random_search(mutation_base=mutation_base, crossover=crossover, selection_base=selection_base,
    n_iterations=n_iterations, n_runs=n_runs, n_generations=n_generations, pop_size=pop_size, elitism=elitism, maximization=maximization,
    verbose_ga=True, mode='summary', seed=None):
    if seed is not None:
        random.seed(seed)

    tried_configs = set()
    results = []

    while len(results) < n_iterations:
        # --- Sample hyperparameters ---
        xo_prob = round(random.uniform(0.6, 1.0), 2)
        mut_prob = round(random.uniform(0.05, 0.4), 2)
        n_swaps = random.randint(2, 10)
        tournament_size = random.randint(2, 10)

        # Check for duplicates
        config_key = (xo_prob, mut_prob, n_swaps)
        if config_key in tried_configs:
            continue
        tried_configs.add(config_key)

        print(f'Configuration {len(results)+1}: xo_prob={xo_prob}, mut_prob={mut_prob}, n_swaps={n_swaps}')
        run_histories = []

        for run in range(n_runs):
            print(f'  Run {run + 1}/{n_runs}')

            # --- Apply partial to mutation operator ---
            mutation = partial(mutation_base, n_swaps=n_swaps)
            selection = partial(selection_base, tournament_size=tournament_size)

            # --- Run GA ---
            population = Population(population_size=pop_size, crossover_function=crossover, mutation_function=mutation)

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

            run_histories.append(fitness_history)

        # --- Store Results ---
        if mode == 'detailed':
            results.append({
                'xo_prob': xo_prob,
                'mut_prob': mut_prob,
                'n_swaps': n_swaps,
                'fitness_history': json.dumps(run_histories)
            })
        else:  # summary: average best fitness across runs
            best_fitnesses = [max(h) if maximization else min(h) for h in run_histories]
            avg_best = sum(best_fitnesses) / n_runs
            results.append({
                'xo_prob': xo_prob,
                'mut_prob': mut_prob,
                'n_swaps': n_swaps,
                'avg_best_fitness': avg_best
            })

    df = pd.DataFrame(results)

    return df