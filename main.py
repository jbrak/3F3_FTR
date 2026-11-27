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
print(f"🔴 OUTPUT PATH: {OUTPUT_PATH}")

##################################################################################################################
# IMPORTING DATA
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

print("🔴 IMPORTING DATA: Successfully Imported Data")

##################################################################################################################
# VALIDATING DATA
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

print("🔴 VALIDATING DATA: Successfully Validated Data")

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

# Total number of datapoints
total = len(data_lst)

symbols_num = len(symbols)

##################################################################################################################
# SECTION 1
# Plot a histogram of each state's frequncies

plt.figure()
plt.bar(symbols.values(), list(count_symbols.values()))
plt.xlabel("Symbol")
plt.ylabel("Count")

plt.savefig(OUTPUT_PATH + "/section_1_histogram.pdf")
print("🟠 SECTION 1 HISTOGRAM: section_1_histogram.pdf")

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

print("🟡 SECTION 2.1 BINOMIAL TAIL FUNCTION: Assertions Passed")

##################################################################################################################
# SECTION 3.1
# Testing the data for IID Uniform

# Assume each symbol is equally likely
p = 1/symbols_num

# Null Hypothesis: We assume a binomial model of B(total, p) for the number of each symbol
# Compute the tail_probabilities for each symbol according to this model
tail_probabilities = {key : binom_tail(item, total, p) for key, item in count_symbols.items()}

# Calculate the required value of t for us to reject the null hypothesis
t_reject = min(tail_probabilities.values()) * 4

# print(p)
# print(tail_probabilities)
# print(t_reject)

with open(OUTPUT_PATH + "/section_3.1_IID_uniform.txt","w") as f:

    str1 = "\n    ".join([f"{str(symbols[key]) + " : " + str(item) + " : " + str(tail_probabilities[key])}" for key, item in count_symbols.items()])

    f.write(f"""
    Null Hypothesis: Each symbol is distributed according to a uniform IID distribution, Nsymbol ~ B(N,p)
    Alternative Hypothesis: Each symbol is not distributed according to a uniform IID distribution
    
    N = {total}
    p = {p}
    
    Symbol : Nsymbol : Tail Probability
    {str1}
    
    Thus the value of t/{symbols_num} to reject H0 = {t_reject/symbols_num}
    Thus the Value of t to reject H0 = {t_reject}
    """)

print(f"🟢 SECTION 3.1 TESTING FOR UNIFORM IID: section_3.1_IID_uniform.txt")

##################################################################################################################
# SECTION 3.2
# Testing for pair dependence

# Null Hypothesis: Consecutive values are independent of one another
# Alternate Hypothesis: Consecutive values are not independent of one another

# Calculate the matrix M, which tracks how many times we transition from state i -> state j

# M_symbols is the matrix represented as a dictionary for easy assignment in the form of:
# M_symbols[i][j] = Mij
M_symbols = {key1 : {key2 : 0 for key2 in symbols.keys()} for key1 in symbols.keys()}

# print(M)

# Create lists to represent the data at the current time step and one time step into the future
data_np_t = np.copy(data_np)[:-1]
data_np_t_p1 = np.copy(data_np)[1:]

# print(data_np)
# print(data_np_t)
# print(data_np_t_p1)

# Generate M_symbols
for i in range(len(data_np_t)):
    M_symbols[data_np_t[i]][data_np_t_p1[i]] += 1

#print(M_symbols)

# Create a function that can be reused in section 3.3
def pair_dependence_test(M_symbols):
    # Numpy Matrix form, rather than dictionary lookup form
    M_matrix = np.zeros((symbols_num, symbols_num))

    # Map the lookup dictionary representation of M to a Numpy array
    for i in range(symbols_num):
        for j in range(symbols_num):
            M_matrix[i,j] = list(list(M_symbols.values())[i].values())[j]

    # Convert floats to integers
    M_matrix = np.int64(M_matrix)

    # print(M_matrix)

    one_vector = np.ones(symbols_num)

    # compute the values of the vectors r and c
    # M_matrix @ one_vector sums up the rows
    r = (M_matrix @ one_vector)/(total-1)
    # one_vector @ M_matrix sums up the columns
    c = (one_vector @ M_matrix)/(total-1)

    # print(r)
    # print(c)
    # print(M_matrix)

    # Create a 0 matrix for tail probabilities
    T_matrix = np.zeros((symbols_num, symbols_num))

    for i in range(symbols_num):
        for j in range(symbols_num):
            T_matrix[i,j] = binom_tail(M_matrix[i,j], total-1, r[i] * c[j])

    # print(T_matrix)
    # print(T_matrix.min())

    # Compute the value of t required to reject the null hypothesis
    t_reject = T_matrix.min() * (symbols_num)**2

    return M_matrix, T_matrix, t_reject

M_matrix, T_matrix, t_reject = pair_dependence_test(M_symbols)

# There is 1 less transition than there is symbols, so this acts as a quick check
assert M_matrix.sum() == total - 1

with open(OUTPUT_PATH + "/section_3.2_dependence.txt", "w") as f:
    f.write(f"""
    Null Hypothesis: Consecutive values are independent of one another
    Alternate Hypothesis: Consecutive values are not independent of one another
    
    Note i and j are iterated in this order: {list(symbols.values())}
    
    M = 
{M_matrix}
    
    T = 
{T_matrix}

    Thus the value of t/{symbols_num**2} to reject H0 = {t_reject / symbols_num**2}
    Thus the Value of t to reject H0 = {t_reject}
    """)

print(f"🟢 SECTION 3.2 TESTING FOR PAIR DEPENDENCE: section_3.2_dependence.txt")
##################################################################################################################
# SECTION 3.3
# Testing for Markovian Property

# Null Hypothesis: Proccess has Markovian Property
# Alternate Hypothesis: Proccess does not have Markovian Property

# Create numpy arrays corresponding to t-1, t and t+1
data_np_t_m1 = np.copy(data_np)[:-2]
data_np_t = np.copy(data_np)[1:-1]
data_np_t_p1 = np.copy(data_np)[2:]

# print(data_np_t_m1)
# print(data_np_t)
# print(data_np_t_p1)

# L_symbols is the matrix represented as a dictionary for easy assignment in the form of:
# L_symbols[xt][i][j] = Lij
L_symbols = {key : {key1 : {key2 : 0 for key2 in symbols.keys()} for key1 in symbols.keys()} for key in symbols.keys()}

# Numpy forms of the matrix L
L_matrices = {}
T_matrices = {}
t_rejects = {}

for i in range(len(data_np_t)):
    L_symbols[data_np_t[i]][data_np_t_m1[i]][data_np_t_p1[i]] += 1

# print(L_symbols)

# Iterate over all the keys and generate the relevent matrices and significance values
for key in symbols.keys():
    L_matrices[key], T_matrices[key], t_rejects[key] = pair_dependence_test(L_symbols[key])
    # print(L_matrices[key], "\n", T_matrices[key],"\n", t_rejects[key])

# Find the minimum value of t required to reject the null hypothesis
t_reject = min(t_rejects.values())*symbols_num

# print(t_reject)

with open(OUTPUT_PATH + "/section_3.3_markov.txt", "w") as f:
    str1 = ""

    for i in symbols.keys():
        str1 += f"""
        Symbol : {symbols[i]}
        ------
        L = 
{L_matrices[i]}

        T = 
{T_matrices[i]}

        t_reject/{symbols_num ** 3} = {t_rejects[i] / symbols_num ** 2}
        t_reject = {t_rejects[i]*symbols_num}
"""

    f.write(f"""
        Null Hypothesis: Proccess has Markovian Property
        Alternate Hypothesis: Proccess does not have Markovian Property

        Note i and j are iterated in this order: {list(symbols.values())}
        {str1}
        Thus the value of t/{symbols_num ** 3} to reject H0 = {t_reject / symbols_num ** 3}
        Thus the Value of t to reject H0 = {t_reject}
        """)


print(f"🟢 SECTION 3.3 TESTING FOR MARKOVIAN PROPERTY: section_3.3_markov.txt")

##################################################################################################################
# SECTION 4.0
# Fitting a Markov Model

Q_matrix = np.float64(np.copy(M_matrix))

# print(Q_matrix)
# print(Q_matrix[:,1])
# print(Q_matrix[1,:])

one_vector = np.ones(symbols_num)

for i in range(symbols_num):
    # print(Q_matrix[:, i] @ one_vector)
    # print(Q_matrix[:, i])
    # print(Q_matrix[:,i] / (Q_matrix[:,i] @ one_vector))

    # Calcualates the model by dividing each entry in the M matrix by the sum of its row
    Q_matrix[i] = Q_matrix[i] / (Q_matrix[i] @ one_vector)

    # print(Q_matrix[:, i])

#print(Q_matrix)

with open(OUTPUT_PATH + "/section_4.0_markov_fit.txt", "w") as f:
    f.write(f"""
    We fit the Markov model by dividing the number of times Xt+1 = j and Xt = i and dividing by the number of times
    that Xt = i, i.e. Qij = Mij / sum over i (Mij)
    
    Note i and j are iterated in this order: {list(symbols.values())}
    
    Fitted Markov Model:
    Q = 
{Q_matrix}
    """)

print(f"🔵 SECTION 4.0 FITTING MARKOV MODEL: section_4.0_markov_fit.txt")
##################################################################################################################
# SECTION 4.1
# Testing Autocorrelation Predictions

# Find the stationary distribution by solving the eigenvector problem
# Stationary distribution is the normalized eigenvalue corresponding to the unit eigenvector
def find_stationary_dist():

    # Compute eigenvalues and eigenvectors
    # Need to transpose the matrix because currently rows indicate prev. state and columns indicate next state
    # We want the opposite
    eig = np.linalg.eig(Q_matrix.T)
    eigenvalues = eig.eigenvalues

    # print(eigenvalues)
    # print(eig.eigenvectors)

    # Identifiy index of unit eigeinvalue, raise an error if one is not present
    for i in range(len(eigenvalues)):
        if np.round(eigenvalues[i],5) == 1:
            index = i
            break
    else:
        raise (ValueError("Transition Matrix does not have a unit eigenvalue"))

    # Identifiy corresponding eigenvector
    stationary = eig.eigenvectors[:,index]
    # print(stationary)

    # Normalize so it has unit sum
    return (stationary / stationary.sum()).real


def autocorrelation(k):
    # Define complex symbols at t = t
    z = np.array([np.exp(1j * 2*np.pi*i/symbols_num) for i in range(symbols_num)])

    # Calculat the stationay distribution
    stationary = find_stationary_dist()

    # Raise Q to the power of k
    Q_k_matrix = np.linalg.matrix_power(Q_matrix, k)

    # Set initially to 0
    r = 0

    # Compute autocorrelation iterativly
    for i in range(symbols_num):
        ej = 0
        for j in range(symbols_num):
            ej += np.conj(z[j]) * Q_k_matrix[i,j]

        r += z[i] * stationary[i] * ej

    return r

# Validation check to ensure it is behaving as expected
assert np.round(autocorrelation(0),10) == 1

# print(find_stationary_dist())
# print(autocorrelation(0))
# print(autocorrelation(1))

# Now lets compute the empirical autocorrelation
# Define a list of different delays
k = range(6)

# Generate the complex symbols
complex_symbols = {i[1] : np.exp(1j * 2*np.pi*i[0]/symbols_num) for i in enumerate(symbols.keys())}

# print(complex_symbols)

# Create an empty array with complex symbols rarther than those from the file
data_np_complex = np.zeros(total, np.complex128)

# print(len(data_np_complex))

# Populate the complex array
for key,item in complex_symbols.items():
    data_np_complex[data_np == key] = item

# print(data_np_complex)

# Create a dictionary to the store E[Xt Xt+k*] for each delay
r_emp_data = {}
r_emp = {}
r_auto = {}

for i in k:

    # Create arrays for data_np and data_np a time step of k in the past
    data_np_t = data_np_complex[:-i] if i > 0 else data_np_complex
    data_np_t_pk = data_np_complex[i:]

    # print(data_np_t_pk)

    # Compute the product of Xt and Xt+k* at each timestep
    r_emp_data[i] = np.conj(data_np_t_pk) * data_np_t

    # Average across the entire time
    r_emp[i] = (r_emp_data[i].sum())/len(r_emp_data[i])

    # print(data_np_t)
    # print(data_np_t_pk)

    # Compute the model's predicted value
    r_auto[i] = autocorrelation(i)

    # plt.figure()
    # plt.scatter(x=range(len(r_emp[i])),y=r_emp[i].real)
    # plt.scatter(x=range(len(r_emp[i])),y=r_emp[i].imag)
    # plt.scatter(range(len(r_emp[i])), abs(r_emp[i]))
    # plt.plot(range(len(r_emp[i])),[r_auto.imag for i in range(len(r_emp[i]))])

    # print(r_emp[i], r_auto)

# Convert from dictionaries to numpy arrays for easy plotting
r_emp_plot = np.array(list(r_emp.values()))
r_auto_plot = np.array(list(r_auto.values()))

# print(np.array(list(r_emp.values())).imag)

# Produce a graph plotting the empirically calcualted autocorrelation against
# the predicted value from the model for diferent valus of k
# Plot Real, Imaginary, Modulous and Argument form for each graph
fig, ax = plt.subplots(2,2)

ax[0,0].scatter(k, r_emp_plot.real, color = "red", label = "Empirical")
ax[0,0].plot(k, r_auto_plot.real, label = "Model")
ax[0,0].set_xlabel("k")
ax[0,0].set_ylabel(r"$\Re \left[ r(k) \right]$")

ax[1,0].scatter(k, r_emp_plot.imag, color = "red")
ax[1,0].plot(k, r_auto_plot.imag)
ax[1,0].set_xlabel("k")
ax[1,0].set_ylabel(r"$\Im \left[ r(k) \right]$")

ax[0,1].scatter(k, abs(r_emp_plot), color = "red")
ax[0,1].plot(k, abs(r_auto_plot))
ax[0,1].set_xlabel("k")
ax[0,1].set_ylabel(r"$\left| r(k) \right|$")

ax[1,1].scatter(k, np.angle(r_emp_plot), color = "red")
ax[1,1].plot(k, np.angle(r_auto_plot))
ax[1,1].set_xlabel("k")
ax[1,1].set_ylabel(r"$ \arg \left[ r(k) \right]$")

fig.tight_layout(pad = 3)
fig.legend(loc ='upper center', ncol = 2)
fig.savefig(OUTPUT_PATH + "/section_4.1_autocorrelation.pdf")

# Write the results to a text file for storage
with open(OUTPUT_PATH + "/section_4.1_autocorrelation.txt", "w") as f:
    full = ""
    real = ""
    imagi = ""
    mod = ""
    argu = ""

    for i in k:
        full += f"{i} : {r_auto[i]} : {r_emp[i]}\n        "
        real += f"{i} : {r_auto[i].real} : {r_emp[i].real}\n        "
        imagi += f"{i} : {r_auto[i].imag} : {r_emp[i].imag}\n        "
        mod += f"{i} : {abs(r_auto[i])} : {abs(r_emp[i])}\n        "
        argu += f"{i} : {np.angle(r_auto[i])} : {np.angle(r_emp[i])}\n        "


    f.write(f"""
        Comparing the Autocorrelation of the model to the empirical value from the data for different timesteps
        
        Full Complex Number:
        k : Prediction from the Model : Empirical Value
        {full}
        Real Component:
        k : Prediction from the Model : Empirical Value
        {real}
        Imaginary Component:
        k : Prediction from the Model : Empirical Value
        {imagi}
        Magnitude:
        k : Prediction from the Model : Empirical Value
        {mod}
        Argument:
        k : Prediction from the Model : Empirical Value
        {argu}
        """)

print(f"🔵 SECTION 4.0 FITTING MARKOV MODEL: section_4.1_autocorrelation[.pdf,.txt]")

##################################################################################################################
plt.show()