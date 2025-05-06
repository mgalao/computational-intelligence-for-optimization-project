"""
This module loads the artist and conflict data.
It reads the 'artists.csv' and 'conflicts.csv' files from the same directory
and provides them as Pandas DataFrames for use in the optimization pipeline.
"""

import pandas as pd
import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the paths to the CSV files
artists_path = os.path.join(script_directory, "artists.csv")
conflicts_path = os.path.join(script_directory, "conflicts.csv")

# Read the CSV files
artists = pd.read_csv(artists_path, index_col=0)
conflicts_matrix = pd.read_csv(conflicts_path, index_col=0)
conflicts_matrix = conflicts_matrix.to_numpy()