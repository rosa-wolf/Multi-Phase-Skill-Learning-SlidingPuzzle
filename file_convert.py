import numpy as np

data = np.load("configuration.npz")
for key, value in data.items():
    np.savetxt(f"configuration_{key}.csv", value, delimiter=',')