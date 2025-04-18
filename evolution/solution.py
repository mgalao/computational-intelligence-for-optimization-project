"""
This module defines the Solution base class and its concrete implementation MLFSolution,
which represents a valid lineup for the problem. It includes methods for initialization, validation,
random generation, and fitness evaluation based on prime slot popularity, genre diversity, and conflict penalty.
"""

from abc import ABC, abstractmethod
import random
from copy import deepcopy
from data.import_data import artists, conflicts_matrix
from itertools import combinations

class Solution(ABC):
    """
    Abstract base class representing a generic solution for an optimization problem.
    Any specific problem must inherit from this and implement the required methods.
    """

    def __init__(self, repr=None):
        """
        Initializes a Solution. If no representation is provided, a random one is generated.
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


class MLFSolution(Solution):
    """
    Concrete implementation of a Solution for the Music Lineup Festival (MLF) problem.
    Represents a schedule assigning each artist to a unique stage and time slot.
    Evaluates fitness based on:
        - Prime Slot Popularity
        - Genre Diversity
        - Conflict Penalty
    """

    def __init__(self, repr=None, num_stages=5, num_slots_per_stage=7):
        """
        Initializes an MLF solution with given number of stages and slots.
        Validates the provided representation, if any.
        """

        # Number of stages and slots per stage
        self.num_stages = num_stages
        self.num_slots_per_stage = num_slots_per_stage
        self.num_slots = num_stages * num_slots_per_stage

        # Validate representation if provided
        if repr:
            self._validate_repr(repr)

        # If no representation is provided, generate a random one
        super().__init__(repr=repr)

    def _validate_repr(self, repr):
        """
        Validates that the representation:
            - Is a list
            - Has exactly one unique artist per slot
            - Contains valid artist IDs
        """

        # Check if representation is a list
        if not isinstance(repr, list):
            raise TypeError("Representation must be a list.")
        
        # Check if representation has the correct number of slots
        if len(repr) != self.num_slots:
            raise ValueError(f"Representation must contain {self.num_slots} elements.")
        
        # Check if all elements are integers (artist IDs)
        if not all(isinstance(artist_id, int) for artist_id in repr):
            raise TypeError("All elements in the representation must be integers.")
        
        # Check if each artist ID is unique
        if len(set(repr)) != len(repr):
            raise ValueError("Each artist must be assigned exactly once.")

        # Check if all artist IDs are valid
        valid_ids = set(artists['artist_id'])
        if not all(artist_id in valid_ids for artist_id in repr):
            raise ValueError("All artist IDs must exist in the dataset.")
    
    def random_initial_representation(self):
        """
        Generates a valid random lineup by shuffling all artist IDs.
        Each artist is scheduled once, across all stages and slots.
        """
        artist_ids = list(artists['artist_id'])
        random.shuffle(artist_ids)
        return artist_ids

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
            row['artist_id']: {'popularity': row['popularity'], 'genre': row['genre']}
            for _, row in artists.iterrows()
        }

        # Calculate individual components of the fitness score
        prime_slot_popularity = self._get_prime_slot_popularity(artist_info)
        genre_diversity = self._get_genre_diversity(artist_info)
        conflict_penalty = self._get_conflict_penalty()

        # Combine components into a final fitness score
        return (prime_slot_popularity + genre_diversity - conflict_penalty) / 3 # Normalize to [0, 1]

    def _get_prime_slot_popularity(self, artist_info):
        prime_slot_indices = []
        for stage in range(self.num_stages):
            prime_slot_indices.append(stage * self.num_slots_per_stage + self.num_slots_per_stage-1) # stage * number_slots_per_stage is the first slot in each stage. we have to add the other slots without forgetting the 0 index
        
        prime_slot_popularity = 0
        for index in prime_slot_indices:
            prime_slot_popularity += artist_info[self.repr[index]]['popularity']

        popularities = [artist['popularity'] for artist in artist_info.values()]
        top_popularities = sorted(popularities, reverse=True)[:self.num_stages]

        maximum_possibile_popularity = 0
        for popularity in top_popularities:
            maximum_possibile_popularity += popularity
        
        normalized_prime_slot_popularity = prime_slot_popularity/maximum_possibile_popularity

        return normalized_prime_slot_popularity
            

    def _get_genre_diversity(self, artist_info):
        possible_genres = []
        for artist in artist_info.values():
            if artist['genre'] not in possible_genres: possible_genres.append(artist['genre'])
        num_possible_genres = len(possible_genres)

        maximum_genre_diversity = min(self.num_stages, num_possible_genres)

        genre_diversity_per_slot = []
        for slot in range(self.num_slots_per_stage):
            genres = []
            for stage in range(self.num_stages):
                idx = stage * self.num_slots_per_stage + slot
                genres.append(artist_info[self.repr[idx]]['genre'])
            
            genre_diversity_per_slot.append(len(set(genres)) / maximum_genre_diversity)
        
        avg_genre_diversity = sum(genre_diversity_per_slot)/ len(maximum_genre_diversity)
        return avg_genre_diversity


    def _get_conflict_penalty(self, artist_info, conflicts_matrix):
        conflicts = []
        for slot in range(self.num_slots_per_stage):
            conflicts_per_slot = []
            artists = []
            for stage in range(self.num_stages):
                idx = stage * self.num_slots_per_stage + slot
                artists.append(self.repr[idx])
            for artist1, artist2 in combinations(artists,2):
                conflicts_per_slot.append(conflicts_matrix[artist1][artist2])
            
            conflicts.append(sum(conflicts_per_slot))

        number_conflicts_per_slot = (self.num_stages * (self.num_stages -1))/2 
        maximum_possible_conflict = (1*number_conflicts_per_slot) * self.num_slots_per_stage

        normalized_conflict = sum(conflicts) / maximum_possible_conflict

        return normalized_conflict


        




                