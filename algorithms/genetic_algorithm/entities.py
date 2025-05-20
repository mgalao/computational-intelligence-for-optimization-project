"""
This module defines the base class 'Solution' and its concrete implementation 'Individual'.
It includes functionality for generating, validating, and evaluating lineup solutions based on
prime slot popularity, genre diversity, and conflict penalties.
A 'Population' class is also defined to manage a collection of unique individuals.
"""

from data.import_data import artists, conflicts_matrix
from algorithms.genetic_algorithm.mutation import n_swap_mutation 
from utils import *

class Solution(ABC):
    """
    Abstract base class representing a generic solution for an optimization problem.
    Any specific problem must inherit from this and implement the required methods.
    """

    def __init__(self, repr=None):
        """
        Initializes a Solution. If no representation is provided,
        a random one is generated.
        """
        if repr is None:
            repr = self.random_initial_representation()
        self.repr = repr

    def __repr__(self):
        """
        String representation of the solution object.
        """
        return str(self.repr)

    @abstractmethod
    def fitness(self):
        """
        Calculates the fitness score of the solution.
        Implemented by subclass.
        """
        pass

    @abstractmethod
    def random_initial_representation(self):
        """
        Generates a valid random representation of a solution.
        Implemented by subclass.
        """
        pass


class Individual(Solution):
    """
    Concrete implementation of a Solution for the Music Lineup Festival (MLF) problem.
    Represents a schedule assigning each artist to a unique stage and time slot.
    Evaluates fitness based on:
        - Prime Slot Popularity
        - Genre Diversity
        - Conflict Penalty
    """
            
    def __init__(self,
                repr=None,
                artists=artists,
                conflicts_matrix=conflicts_matrix,
                mutation_function=None,
                crossover_function=None,
                ):
            """
            Initializes an Individual with a representation (if provided)
            and validates it.
            """

            # Set the artists and conflicts matrix
            self.artists = artists
            self.conflicts_matrix = conflicts_matrix
            self.mutation_function = mutation_function
            self.crossover_function = crossover_function

            # If a representation is provided, infer dimensions from it
            if repr:
                self.num_stages = len(repr[0]) # cols
                self.num_slots_per_stage = len(repr) # rows
                self.num_slots = self.num_stages * self.num_slots_per_stage
                self._validate_repr(repr)
            else:
                # If no representation is provided, set default dimensions
                self.num_stages = 5
                self.num_slots_per_stage = 7
                self.num_slots = self.num_stages * self.num_slots_per_stage

                # Generate a random valid representation
                repr = self.random_initial_representation()

            # Call the superclass constructor
            super().__init__(repr=repr)
    
    def __repr__(self):
        """
        Returns a string representation of the individual as a slot x stage matrix
        and its fitness score.
        """
    
        # Create a string representation of the matrix
        matrix_str = "\n".join(
            [f"Slot {i+1}: {list(row)}" for i, row in enumerate(self.repr)]
        )

        return f"Fitness: {self.fitness():.4f}\n{matrix_str}"
    
    def _validate_repr(self, repr):
        """
        Validates that the representation:
            - Is a list
            - Has exactly one unique artist per slot
            - Contains valid artist IDs
        """

        # Flatten the representation
        flat = [artist for row in repr for artist in row]

        # Check if representation is a matrix
        if not (isinstance(repr, list) and all(isinstance(row, list) for row in repr)):
            raise TypeError("Representation must be a matrix (list of lists).")

        # Check if representation has the correct number of slots
        if len(repr) != self.num_slots_per_stage:
            raise ValueError(f"Representation must contain {self. num_slots_per_stage} slots.")
        
        # Check if all elements are integers (artist IDs)
        if not all(isinstance(artist_id, int) for artist_id in flat):
            raise TypeError("All elements in the representation must be integers.")
        
        # Check if each artist is assigned exactly once
        if len(set(flat)) != len(flat):
            raise ValueError("Each artist must be assigned exactly once.")

        # Check if all artist IDs are valid
        valid_ids = set(self.artists.index)
        if not all(artist_id in valid_ids for artist_id in flat):
            raise ValueError("All artist IDs must exist in the dataset.")
    
    def random_initial_representation(self):
        """
        Generates a valid random lineup by shuffling all artist IDs.
        Each artist is scheduled once, across all stages and slots.
        """

        # Shuffle artist IDs
        artist_ids = list(self.artists.index)
        random.shuffle(artist_ids)

        # Create a matrix of size (num_slots_per_stage x num_stages)
        matrix = [
            artist_ids[i*self.num_stages: (i+1)*self.num_stages] for i in range(self.num_slots_per_stage)
            ]

        return matrix

    def fitness(self):
        """
        Computes the fitness score by combining:
            - Prime Slot Popularity (normalized)
            - Genre Diversity (normalized)
            - Conflict Penalty (normalized and subtracted)
        All components contribute equally to the final fitness.
        """

        # Create a dictionary mapping artist IDs to their popularity and genre
        artist_info = {
            idx: {'popularity': row['popularity'], 'genre': row['genre']}
            for idx, row in self.artists.iterrows()
        }

        # Calculate individual components of the fitness score
        prime_slot_popularity = self.get_prime_slot_popularity(artist_info)
        genre_diversity = self.get_genre_diversity(artist_info)
        conflict_penalty = self.get_conflict_penalty(self.conflicts_matrix)

        # Combine components into a final fitness score
        return (prime_slot_popularity + genre_diversity - conflict_penalty)
    
    def get_prime_slot_popularity(self, artist_info):
        """
        Calculates the prime slot popularity of artists in the lineup.
        The prime slot is the last slot of each stage.
        The popularity is normalized by the maximum possible popularity.
        The maximum possible popularity is the sum of the top N popularities,
        where N is the number of stages.
        """
        
        # Get the last slot of each stage
        prime_slot_artists = self.repr[-1]

        # Calculate the total popularity of artists in prime slots
        prime_slot_popularity = sum(artist_info[artist]['popularity'] for artist in prime_slot_artists)

        # Get the top popularities
        popularities = [artist['popularity'] for artist in artist_info.values()]
        top_popularities = sorted(popularities, reverse=True)[:self.num_stages]

        # Calculate the maximum possible popularity
        maximum_possible_popularity = sum(top_popularities)
        
        # Normalize the prime slot popularity
        normalized_prime_slot_popularity = prime_slot_popularity / maximum_possible_popularity

        return normalized_prime_slot_popularity
            
    def get_genre_diversity(self, artist_info):
        """
        Calculates the genre diversity of the artists in the lineup.
        The genre diversity is the number of unique genres in the lineup,
        normalized by the maximum possible genre diversity.
        The maximum possible genre diversity is the minimum of the number of stages
        and the number of unique genres in the dataset.
        """

        # Get the genre diversity of the artists in the lineup
        possible_genres = []
        for artist in artist_info.values():
            if artist['genre'] not in possible_genres:
                possible_genres.append(artist['genre'])
        num_possible_genres = len(possible_genres)

        # Calculate the maximum possible genre diversity
        maximum_genre_diversity = min(self.num_stages, num_possible_genres)

        # Calculate the genre diversity for each slot
        genre_diversity_per_slot = []
        for slot in range(self.num_slots_per_stage):
            genres = []
            for stage in range(self.num_stages):
                idx = self.repr[slot][stage]
                genres.append(artist_info[idx]['genre'])
            
            # Calculate the genre diversity for this slot (normalized)
            genre_diversity_per_slot.append(len(set(genres)) / maximum_genre_diversity)
        
        # Calculate the average genre diversity across all slots
        avg_genre_diversity = sum(genre_diversity_per_slot) / len(genre_diversity_per_slot)

        return avg_genre_diversity
    
    def get_conflict_penalty(self, conflicts_matrix):
        """
        Calculates the conflict penalty for the lineup.
        The penalty is based on the number of conflicts between artists in the same slot.
        The penalty is normalized by the maximum possible conflict.
        The maximum possible conflict is the sum of the top cN conflicts,
        where N is the number of slots in a stage.
        """

        # Calculate the number of conflicts for each slot
        number_conflicts_per_slot = int(self.num_stages * (self.num_stages-1) / 2) 

        # Create a mask to get the upper triangle of the conflicts matrix
        mask = np.triu(np.ones(conflicts_matrix.shape, dtype=bool), k=1)
        total_conflicts = conflicts_matrix[mask].tolist()

        # Get the top conflicts
        top_conflicts = sorted(total_conflicts, reverse=True)[:number_conflicts_per_slot]

        # Calculate the maximum possible conflict
        maximum_conflict_per_slot = sum(top_conflicts)

        # Calculate the conflict penalty for each slot
        conflicts = []
        for slot in range(self.num_slots_per_stage):
            conflicts_per_slot = []
            artists_idx = []

            # Get the artists indexes for this slot
            for stage in range(self.num_stages):
                idx = self.repr[slot][stage]
                artists_idx.append(idx)
            
            # Calculate the conflicts for this slot
            for artist1_idx, artist2_idx in combinations(artists_idx, 2):
                conflicts_per_slot.append(conflicts_matrix[artist1_idx][artist2_idx])
            
            total_slot_conflict = sum(conflicts_per_slot)

            # Normalize
            normalized_conflict = total_slot_conflict / maximum_conflict_per_slot

            # Append the conflict penalty for this slot
            conflicts.append(normalized_conflict)

        return sum(conflicts) / len(conflicts)
    
    def get_slot_fitness(self, slot_idx):
        """
        Computes the fitness of a single slot (row) by combining genre diversity and conflict penalty 
        Note: Prime slot popularity is excluded, since it applies only to the last slot globally.

        This is useful for local evaluation during crossover strategies (EBC).
        
        Args:
            slot_idx (int): Index of the slot to evaluate.

        Returns:
            float: Normalized fitness score for the slot.
        """

        # artist info
        artist_info = {
            idx: {'popularity': row['popularity'], 'genre': row['genre']}
            for idx, row in self.artists.iterrows()
        }

        slot_artists = self.repr[slot_idx]
        genres = [artist_info[artist]['genre'] for artist in slot_artists]
        num_unique_genres = len(set(genres))

        # Calculate maximum possible genre diversity
        all_genres = set(self.artists['genre'])
        max_genre_diversity = min(self.num_stages, len(all_genres))
        normalized_genre_diversity = num_unique_genres / max_genre_diversity

        # Conflict penalty
        number_conflict = int(self.num_stages * (self.num_stages - 1) / 2)
        mask = np.triu(np.ones(conflicts_matrix.shape, dtype=bool), k=1)
        total_conflicts = conflicts_matrix[mask].tolist()
        maximum_possible_conflict = sum(sorted(total_conflicts, reverse=True)[:number_conflict])

        conflicts = []
        artists_idx = []
        
        # Get the artists indexes for this slot
        for stage in range(self.num_stages):
            idx = self.repr[slot_idx][stage]
            artists_idx.append(idx)
            
        # Calculate the conflicts for this slot
        for artist1_idx, artist2_idx in combinations(artists_idx, 2):
            conflicts.append(conflicts_matrix[artist1_idx][artist2_idx])
            
        normalized_conflict_penalty = sum(conflicts) / maximum_possible_conflict 

        # Final fitness 
        slot_fitness = (normalized_genre_diversity - normalized_conflict_penalty) 

        return slot_fitness

    def crossover(self, other_solution):
        """
        Applies the crossover operator to generate two offspring.
        """
        
        # Apply crossover
        offspring1_repr, offspring2_repr = self.crossover_function(self, other_solution)

        return (
            Individual(
                repr=offspring1_repr,
                artists=self.artists,
                conflicts_matrix=self.conflicts_matrix,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
            ),
            Individual(
                repr=offspring2_repr,
                artists=self.artists,
                conflicts_matrix=self.conflicts_matrix,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
            )
        )
    
    def mutation(self, mut_prob):
        """
        Applies the mutation operator to the individual.
        """

        # Apply mutation
        new_repr = self.mutation_function(self, mut_prob)
        
        return Individual(
            repr=new_repr,
            artists=self.artists,
            conflicts_matrix=self.conflicts_matrix,
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
        )
    
    def get_random_neighbor(self):
        """
        Returns a random neighbor by applying a 2-swap mutation to the current individual.
        """

        mutated_repr = n_swap_mutation(self, mut_prob=1, n_swaps=2)
        try:
            return Individual(
                repr=mutated_repr,
                artists=self.artists,
                conflicts_matrix=self.conflicts_matrix,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function
            )
        except Exception:
            return deepcopy(self)  # fallback if mutation is invalid


    def get_neighbors(self):
        """
        Generates 5 valid neighbors by applying 2-swap mutations.
        Skips invalid individuals and retries until 5 are collected or a max number of attempts is reached.
        """

        neighbors = []
        max_attempts = 20  # prevents infinite loops if many mutations are invalid

        while len(neighbors) < 5 and max_attempts > 0:
            mutated_repr = n_swap_mutation(self, mut_prob=1, n_swaps=2)
            try:
                neighbor = Individual(
                    repr=mutated_repr,
                    artists=self.artists,
                    conflicts_matrix=self.conflicts_matrix,
                    mutation_function=self.mutation_function,
                    crossover_function=self.crossover_function
                )
                neighbors.append(neighbor)
            except Exception:
                pass
            max_attempts -= 1

        return neighbors


class Population:
    def __init__(self,
                 population_size,
                 crossover_function,
                 mutation_function
                 ):
        """
        Initializes a population of unique individuals.
        Each individual is represented as a slot x stage matrix.
        The population is generated by creating random individuals
        until the desired population size is reached.
        """

        # Set the population size and initialize the list of individuals
        self.population_size = population_size
        self.individuals = []
        unique_reprs = set() # To check for uniqueness

        # Generate unique individuals until the population size is reached
        while len(self.individuals) < population_size:
            # Create a new individual
            indiv = Individual(
                crossover_function=crossover_function,
                mutation_function=mutation_function
            )

            # Store the representation as a tuple of tuples (to check for uniqueness)
            repr_as_tuple =  tuple(tuple(row) for row in deepcopy(indiv.repr))

            # If the representation is unique, add the individual to the population
            if repr_as_tuple not in unique_reprs:
                self.individuals.append(indiv)
                unique_reprs.add(repr_as_tuple)
    
    def __repr__(self):
        """
        Returns a string representation of the population.
        Each individual is represented as a slot x stage matrix.
        """      
        return "\n\n".join([f"Individual {i}:\n{indiv}" for i, indiv in enumerate(self.individuals)])
    
    def __getitem__(self, index):
        """
        Returns the individual at the specified index.
        """
        return self.individuals[index]

    def __len__(self):
        """
        Returns the number of individuals in the population.
        """
        return len(self.individuals)

    def __iter__(self):
        """
        Returns an iterator over the individuals in the population.
        """
        return iter(self.individuals)

    def best_individuals(self, n=10):
        """
        Returns the n best individuals in the population
        based on highest fitness score.
        """
        
        # Sort the individuals by fitness in descending order
        sorted_individuals = sorted(self.individuals, key=lambda indiv: indiv.fitness(), reverse=True)

        # Get the top n best individuals    
        best_n = sorted_individuals[:n]

        return best_n