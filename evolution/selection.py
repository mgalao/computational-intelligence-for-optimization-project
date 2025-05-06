import random
from copy import deepcopy
from evolution.entities import Solution
from evolution.entities import Individual
import os
import sys
import random

# Add the parent directory to the system path to import the import_data module
parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_folder)

from data import import_data

# Import data from the import_data module
artists = import_data.artists
conflicts_matrix = import_data.conflicts_matrix

def fitness_proportionate_selection(population):

    fitnesses = []
    probabilities = []

    for indiv in population:  fitnesses.append(indiv.fitness())
    total_fitness = sum(fitnesses)

    for indiv in population: probabilities.append(indiv.fitness()/total_fitness)

    selected_indiv = random.choices(population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)

def ranking_selection(population):
    probabilities = []
    sorted_population = sorted(population, key=lambda indiv: indiv.fitness(), reverse=True)

    ranking_position = range(1,len(population)+1)

    for position in ranking_position: probabilities.append(1-(position/sum(ranking_position)))

    selected_indiv = random.choices(sorted_population, weights=probabilities, k=1)[0]

    return deepcopy(selected_indiv)


def tournament_selection(population, tournament_size):
    tournament_population = random.sample(population, tournament_size)
    best_indiv = max(tournament_population, key=lambda indiv: indiv.fitness(), reverse=True)
    return deepcopy(best_indiv)









    

    







    







