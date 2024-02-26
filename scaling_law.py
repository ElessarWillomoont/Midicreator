TOKEN_SIZE = 22346648

# Approach 1 numbers
# # parameters, tokens
# raw = [
#     [400e6, 8e9],
#     [1e9, 20.2e9],
#     [10e9, 205.1e9],
#     [67e9, 1.5e12],
#     [175e9, 3.7e12],
#     [280e9, 5.9e12],
#     [520e9, 11e12],
#     [1e12, 21.2e12],
#     [10e12, 216.2e12],
# ]

# Approach 2 numbers
# parameters, tokens
raw = [
    [400e6, 7.7e9],
    [1e9, 20.0e9],
    [10e9, 219.5e9],
    [67e9, 1.7e12],
    [175e9, 4.3e12],
    [280e9, 7.1e12],
    [520e9, 13.4e12],
    [1e12, 26.5e12],
    [10e12, 292.0e12],
]

# fit a line by linear regression to the raw data
import numpy as np
x = np.array([np.log10(x[0]) for x in raw])
y = np.array([np.log10(x[1]) for x in raw])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"y = {m}x + {c}")

import matplotlib.pyplot as plt

plt.figure(figsize=(3, 3))
# plot the line
plt.plot([q[0] for q in raw], [10**(m*np.log10(q[0]) + c) for q in raw], label='linear regression', color='r')
# plot the raw data
plt.scatter([q[0] for q in raw], [q[1] for q in raw], label='raw data')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('parameters')
plt.ylabel('tokens')
plt.title('compute optimal models')
plt.grid()

xquery = 7000000 # query model size here (e.g. GPT-2 small is 124M)
yquery = 10**(m*np.log10(xquery) + c)
print(f"predicted parameters for {xquery:,.2f} tokens: {yquery:,.2f}")

# Change to predict model size
# fit a line by linear regression to the raw data
import numpy as np

x = np.array([np.log10(x[1]) for x in raw])  # change to tokens
y = np.array([np.log10(x[0]) for x in raw])  # change to parameters
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"y = {m}x + {c}")

# plot the line and raw data
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot([q[1] for q in raw], [10**(m*np.log10(q[1]) + c) for q in raw], label='linear regression', color='r')
plt.scatter([q[1] for q in raw], [q[0] for q in raw], label='raw data')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('tokens')
plt.ylabel('parameters')
plt.title('compute optimal models')
plt.grid()

# predict parameters for a given number of tokens
query_tokens = 174519643  # query token size here (e.g. GPT-2 small is 124M)
query_params = 10**(m*np.log10(query_tokens) + c)
print(f"predicted parameters for {query_tokens:,.0f} model size: {query_params:,.0f}")