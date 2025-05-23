"""
This module implements a genetic algorithm for optimizing a population of solutions.
It includes functions for selection, crossover, mutation, and elitism.
"""

from algorithms.genetic_algorithm.entities import *
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
    

def diversify_population(population, percentage=0.5):
    """
    Injects diversity by replacing the weakest 50% individuals in a Population.
    """

    num_to_replace = int(len(population) * percentage)

    # Get a sorted list of individuals by fitness
    sorted_inds = sorted(population, key=lambda indiv: indiv.fitness(), reverse=True)

    # Access internal list safely (not assuming population is a list)
    individuals = population._Population__dict__["individuals"] if hasattr(population, "_Population__dict__") else population.__dict__["individuals"]

    # Replace the weakest individuals at the end
    for i in range(1, num_to_replace + 1):
        new_individual = Individual(
            crossover_function=sorted_inds[0].crossover_function,
            mutation_function=sorted_inds[0].mutation_function
        )
        individuals[-i] = new_individual



def genetic_algorithm(
    initial_population: 'Population',
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = True,
    xo_prob: float = 0.9,
    mut_prob: float = 0.1,
    elitism: bool = True,
    verbose_ga: Optional[str] = False,
    adapt_on_stable: bool = False,
    params: Optional[dict] = None,
    stability_window: int = 5,
    stability_epsilon: float = 1e-3,
    diversify_on_plateau: bool = False,
) -> tuple[list[float], 'Individual']:
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
        diversify_on_plateau (bool, optional): If True, injects random diversity into the population when a fitness plateau is detected 
                                               and adaptation is disabled. This can help escape local optima and maintain diversity. Defaults to False.


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
        if verbose_ga in ("minimal", "full", True):
            print(f'\n-------------- Generation: {gen} --------------')

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

            if verbose_ga in ("full", True):
                print(f'\nSelected individuals:\n{first_ind}\n{second_ind}\n')
            
            # Choose an operator between crossover and replication
            # Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
                if verbose_ga in ("full", True):
                    print(f'Applied crossover - offspring:')
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
                if verbose_ga in ("full", True):
                    print(f'Applied replication - offspring:')

            if verbose_ga in ("full", True):
                print(f'{offspring1}\n{offspring2}')

            # Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob)
            
            # Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose_ga in ("full", True):
                print(f'\nFirst mutated individual:\n{first_new_ind}')

            if offspring2 is not None and len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose_ga in ("full", True):
                    print(f'Second mutated individual:\n{second_new_ind}')

        # Replace P with P'
        population.__dict__["individuals"] = new_population
        best_fitness = get_best_ind(population, maximization).fitness()
        fitness_history.append(best_fitness)

        if verbose_ga in ("minimal", "full", True):
            print(f'Best fitness: {best_fitness:.4f}')

        if diversify_on_plateau and gen > stability_window and adapt_on_stable==False:
            if verbose_ga in ("minimal", "full", True):
                print("injecting diversity...")
            
            diversify_population(population)

        # EXTRA: Adaptation
        if adapt_on_stable and gen > stability_window:
            recent = fitness_history[-stability_window:]
            if np.std(recent) < stability_epsilon and (gen - last_adaptation_gen) >= adaptation_cooldown:
                if verbose_ga in ("minimal", "full", True):
                    print("Fitness plateau detected — adapting parameters...")

                # xo_prob = min(1.0, xo_prob + 0.025)
                # mut_prob = min(1.0, mut_prob + 0.025)
                if verbose_ga in ("minimal", "full", True):
                    # print(f"adapting probabilities... xo_prob to {xo_prob:.2f}, mut_prob to {mut_prob:.2f}")
                    # print(f" xo_prob to {xo_prob:.2f}, mut_prob to {mut_prob:.2f}")
                    print("adapting parameters...")

                for key in params.get("adapt_keys", []):
                    current = params[key]
                    direction = params["adapt_directions"].get(key, "up")
                    max_key = f"max_{key}"
                    min_key = f"min_{key}"

                    if direction == "up" and max_key in params:
                        params[key] = min(current + 1, params[max_key])
                    elif direction == "down" and min_key in params:
                        params[key] = max(current - 1, params[min_key])


                if verbose_ga in ("minimal", "full", True):
                    values = [f"{k}={params[k]}" for k in params.get("adapt_keys", []) if k in params]
                    print("Adapted parameters: " + ", ".join(values))

                # Re-wrap mutation and selection using updated params
                mutation_function = partial(params["mutation_base"], **{k: params[k] for k in params["mutation_args"]})
                selection_function = partial(params["selection_base"], **{k: params[k] for k in params["selection_args"]})

                # Update mutation function of all individuals
                for ind in population:
                    ind.mutation_function = mutation_function

                # Update selection function used in next generation
                selection_algorithm = selection_function

                last_adaptation_gen = gen

                if diversify_on_plateau:
                    if verbose_ga in ("minimal", "full", True):
                        print("injecting diversity...")
                    
                    diversify_population(population)

    return fitness_history, get_best_ind(population, maximization)