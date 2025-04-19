import os
import sys

# Add the parent directory to the system path to import the import_data module
parent_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_folder)

from data import import_data

# Import data from the import_data module
artists = import_data.artists
conflicts_matrix = import_data.conflicts_matrix

class Individual:
     


class Population:
    ...