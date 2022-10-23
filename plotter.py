import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/nicolas/Desktop/Etudes/4_centrale_sup/CentraleSupelec/cours/centralesup_S_1/lab_project/models/GridWorld/Graph data/Part 1 - Complete Task/Run 2"
timesteps = np.loadtxt(file_path + "/Episode_time")

plt.plot(timesteps)
plt.xlabel("Number of episodes")
plt.ylabel("Timesteps required")

plt.show()
