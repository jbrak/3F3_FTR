##################################################################################################################
# Library Imports

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import math

##################################################################################################################

RUNTIME = int(time.time())
OUTPUT_PATH = f"outputs/{RUNTIME}"
DATA_PATH = f"raw.txt"

##################################################################################################################
# Create filepath to store the figures in

Path(OUTPUT_PATH).mkdir()
print(f"Output path: {OUTPUT_PATH}")

##################################################################################################################
# Extract raw data from the file

with open(DATA_PATH,"r") as f:

    # Read the file and split it into prefix and main body of data
    raw = f.read().split("*")

    #print(raw)

    # Extract the symbol mapping from the prefix
    symbols = {i.split(":")[0] : i.split(":")[1] for i in raw[0].split(";")}

    # Convert main body into a list of data symbols
    data = raw[1].split('\n')

    # Remove first empty string if present
    data = data[1:] if data[0] == '' else data

    #print(data)
    #print(symbols)

##################################################################################################################
# Validate the data set and symbols

count = 0
for i in symbols.keys():
    # Count the number of symbols that match i
    count_i = data.count(i)

    # If less than 1 instance of any given symbol is present warn the user that this is the case
    if count_i < 1:
        print(f"No Instances of Symbol: {i} is present.")

    count += data.count(i)

# Raise an error if the dataset contains a symbol that is not in the prefix. This suggests an invalid symbol
if count != len(data):
    raise (ValueError(f"""
    Number of instances of symbols in prefix does not match total number of symbols in the data set
    There is a symbol not present in the prefix
"""))

##################################################################################################################
# Produce the array in different data structures for convenience

# Produce numpy array version
data_np = np.array(data)
data_lst = data

# Convert to integers
data_int = np.int64(data_np)

# Produce counts of each symbol present in the data
# Key is the symbol as present in data object not the mapping
count_symbols = {i : data_lst.count(i) for i in symbols.keys()}

##################################################################################################################
# SECTION 1
# Plot a histogram of each state's frequncies

plt.figure()
plt.bar(symbols.values(), list(count_symbols.values()))
plt.xlabel("Symbol")
plt.ylabel("Count")

plt.savefig(OUTPUT_PATH + "/section_1_histogram.pdf")
print("🟢 SECTION 1 HISTOGRAM: section_1_histogram.pdf")

##################################################################################################################
# SECTION 2.1
# Define tail probability function for the Binomial Distribution

def binom_tail(x:np.int64, n:np.int64, p:np.float64) -> np.float64:

    # Validate inputs
    if (p>1) or (p<0):
        raise ((ValueError("p must be 0≤p≤1 for it to be a valid probability")))

    if n <= 0:
        raise ((ValueError("n must be greater than 0 for it to be a valid number of trials")))

    if x > n:
        raise (ValueError("x must be less than or equal to n for a valid Binomial Distribution"))

    elif x < 0:
        raise (ValueError("x must be greater than or equal to 0 for a valid Binomial Distribution"))


    # Define a function for the probability mass function
    def pmf(x, n=n, p=p):

        # Calculate n Choose x
        nCx = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))

        return nCx * p ** x * (1 - p) ** (n - x)

    # Define a variable to track 1 - tail probability
    tail_probability_minus_1 = 0

    # Calculate 1 - tail probability by iterating from 0 up to x-1
    for i in range(x):
        tail_probability_minus_1 += pmf(i, n, p)

    # Calculate actual tail probability
    return 1 - tail_probability_minus_1

# Validate the binom_tail function using assertions of simple distributions
assert binom_tail(0,588, 0.24521) == 1.0
assert binom_tail(0,2,0.5) == 1.0
assert binom_tail(1,2,0.5) == 0.75
assert binom_tail(2,2,0.5) == 0.25

print("🟠 SECTION 2.1 BINOMIAL TAIL FUNCTION: Assertions Passed")

##################################################################################################################
plt.show()