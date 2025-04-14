from abc import ABC, abstractmethod

class Solution(ABC):
    def __init__(self, repr=None):
        # To initialize a solution we need to know it's representation. If no representation is given, a solution is randomly initialized.
        if repr == None:
            repr = self.random_initial_representation()
        # Attributes
        self.repr = repr

    # Method that is called when we run print(object of the class)
    def __repr__(self):
        return str(self.repr)

    # Other methods that must be implemented in subclasses
    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass
