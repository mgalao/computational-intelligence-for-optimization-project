"""
This module implements a genetic algorithm for optimizing a population of solutions.
It includes functions for selection, crossover, mutation, and elitism.
"""

from evolution.entities import *
from utils import *


def get_best_ind(
    population: list[Solution], 
    maximization: bool
) -> 'Individual':
    """
    Returns the best individual from the population based on fitness.
    """
    
    # Get the fitness of each individual in the population
    fitness_list = [ind.fitness() for ind in population]

    # If maximization is True, return the individual with the highest fitness
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    # If maximization is False, return the individual with the lowest fitness
    else:
        return population[fitness_list.index(min(fitness_list))]

def genetic_algorithm(
    initial_population: 'Population',
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = True,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose_ga: bool = True
) -> 'Individual':
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (Population): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to True.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """

    # Initialize a population with N individuals
    population = initial_population
    fitness_history = []
    

    # Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose_ga:
            print(f'-------------- Generation: {gen} --------------')

        # Create an empty population P'
        new_population = []

        # If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(deepcopy(get_best_ind(population, maximization)))
        
        # Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population)
            second_ind = selection_algorithm(population)
            
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
            first_new_ind = offspring1.mutation(mut_prob)

            # Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose_ga:
                print(f'\nFirst mutated individual:\n{first_new_ind}')

            if offspring2 is not None and len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose_ga:
                    print(f'Second mutated individual:\n{second_new_ind}')
            
        
        # Replace P with P'

        population = new_population


        if verbose_ga:
            print(f'\nFinal best individual in generation: {get_best_ind(population, maximization).fitness():.4f}\n')
        
        fitness_history.append(get_best_ind(population, maximization).fitness())


    # Return the best individual in P
    return fitness_history, get_best_ind(population, maximization)