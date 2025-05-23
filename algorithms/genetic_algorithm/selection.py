"""
This module contains various selection functions for choosing individuals from a population
in a genetic algorithm. The selection methods include:
1. fitness_proportionate_selection: Selects individuals based on their fitness proportion.
2. ranking_selection: Selects individuals based on their rank in the population.
3. tournament_selection: Selects the best individual from a randomly chosen subset of the population.
"""

from algorithms.genetic_algorithm.entities import *
from utils import *

def fitness_proportionate_selection(
    population: 'Population'
) -> 'Individual':
    """
    Selects an individual from the population based on fitness proportionate selection.
    
    Args:
        population (Population): The population from which to select an individual.
        maximization (bool): If True, maximizes the fitness function; otherwise, minimizes.
    Returns:
        Individual: The selected individual.
    """

    # Calculate the fitness of each individual in the population
    fitnesses = []
    probabilities = []

    for indiv in population:
        fitnesses.append(indiv.fitness())

    # Shift fitness if there are negative or zero values
    min_fitness = min(fitnesses)
    if min_fitness <= 0:
        shift = abs(min_fitness) + 1e-6
        fitnesses = [f + shift for f in fitnesses]

    total_fitness = sum(fitnesses)

    for f in fitnesses:
        probabilities.append(f / total_fitness)

    selected_indiv = random.choices(population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)

def ranking_selection(population):
    """
    Selects an individual from the population using ranking-based selection.

    Individuals are sorted by fitness, and selection probabilities are assigned
    according to their rank (not raw fitness), giving higher-ranked individuals
    a greater chance of being selected.

    Args:
        population (Population): The population from which to select an individual.

    Returns:
        Individual: A deep copy of the selected individual.
    """

    # Sort individuals by descending fitness
    sorted_population = sorted(population, key=lambda indiv: indiv.fitness(), reverse=True)

    # Assign ranks (1 to N) and compute corresponding weights
    max_rank = len(population)
    ranking_position = range(1, max_rank + 1)
    raw_weights = [max_rank - pos + 1 for pos in ranking_position]

    total = sum(raw_weights)
    probabilities = [w / total for w in raw_weights]

    # Randomly select one individual according to the rank-based probabilities
    selected_indiv = random.choices(sorted_population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)

def tournament_selection(population, tournament_size=4):
    """
    Selects an individual from the population using tournament selection.

    A random subset of individuals is sampled, and the one with the highest
    fitness is selected. This method introduces selection pressure based on
    the size of the tournament.

    Args:
        population (Population): The population from which to select an individual.
        tournament_size (int, optional): Number of individuals competing in the tournament. Defaults to 4.

    Returns:
        Individual: A deep copy of the selected individual.
    """

    # Randomly sample a subset of the population
    tournament_population = random.sample(list(population), tournament_size)

    # Select the best individual in the tournament
    best_indiv = max(tournament_population, key=lambda indiv: indiv.fitness())

    return deepcopy(best_indiv)