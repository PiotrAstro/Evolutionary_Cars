import numpy as np
from matplotlib import pyplot as plt

FILE = "example_rewards_for_trajectory.txt"
SAVE_FILE = "example_rewards_for_trajectory.png"

# Load the data
data = np.loadtxt(FILE)

# Plot the data
plt.figure(figsize=(12, 10))
plt.scatter(np.arange(len(data)), data, marker="o", s=1, c="red", label="Reward at each timestep")
plt.plot(np.cumsum(data), label="Cumulative reward", c="blue")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.yscale("symlog")
plt.title("Rewards for Mutation-Only Genetic Algorithm example environment run")
plt.legend()
plt.grid()
plt.savefig(SAVE_FILE)