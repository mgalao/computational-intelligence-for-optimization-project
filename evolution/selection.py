"""
This module contains various selection functions for choosing individuals from a population
in a genetic algorithm. The selection methods include:
1. fitness_proportionate_selection: Selects individuals based on their fitness proportion.
2. ranking_selection: Selects individuals based on their rank in the population.
3. tournament_selection: Selects the best individual from a randomly chosen subset of the population.
"""

from evolution.entities import *
from copy import deepcopy
import random

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
    total_fitness = sum(fitnesses)

    for indiv in population:
        probabilities.append(indiv.fitness()/total_fitness)

    selected_indiv = random.choices(population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)

def ranking_selection(population):
    probabilities = []
    sorted_population = sorted(population, key=lambda indiv: indiv.fitness(), reverse=True)

    ranking_position = range(1,len(population)+1)

    for position in ranking_position: probabilities.append(1-(position/sum(ranking_position)))

    selected_indiv = random.choices(sorted_population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)

def tournament_selection(population, tournament_size=4):

    tournament_population = random.sample(list(population), tournament_size)
    best_indiv = max(tournament_population, key=lambda indiv: indiv.fitness())

    return deepcopy(best_indiv)