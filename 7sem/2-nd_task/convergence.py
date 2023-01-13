import numpy as np
import matplotlib.pyplot as plt

d = 1
NStart = 200
TIME = 1.5
courant = 0.5

rmin, rmax = 0, 1.8
c = 1.5
h = (rmax - rmin) / NStart
tau = courant * h / c
T = int(TIME / tau)
print(T)
# x_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)

with open(f"2-nd_task/d{d}_N{NStart}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U200 = np.loadtxt(file)

with open(f"2-nd_task/d{d}_N{NStart * 3}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U600 = np.loadtxt(file)

with open(f"2-nd_task/d{d}_N{NStart * 9}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U1800 = np.loadtxt(file)

print(f"U200 shape = {np.shape(U200[:, 1:-1])}")
print(f"U600 shape = {np.shape(U600[:-1:3, 2::3])}")
print(f"U1800 shape = {np.shape(U1800[:-1:9, 5::9])}")

T = 499
conv_C = np.zeros(T + 1)
conv_L2 = np.zeros(T + 1)
for n in range(T + 1):
    conv_C[n] = np.max(np.abs(U600[3*n, 2::3] - U200[n, 1:-1])) / \
        np.max(np.abs(U600[3*n, 2::3] - U1800[9*n, 5::9]))
    # removed h 
    conv_L2[n] = np.sqrt(np.sum((U600[3*n, 2::3] - U200[n, 1:-1])**2)) / \
        np.sqrt(np.sum((U600[3*n, 2::3] - U1800[9*n, 5::9])**2))

fig = plt.figure()
# fig.set_label(f"O2 grid convergence for 1D WE in R{d}")
ax = plt.subplot()
time_grid = np.linspace(0, 1.5, T + 1)
ax.plot(time_grid, conv_C, "g", label="C-norm")
ax.plot(time_grid, conv_L2, "b", label="L2-norm")
ax.plot(time_grid, 9*np.ones(T + 1), "r", label="Ref=9")
ax.set_xlabel("Time")
ax.set_ylabel("Ratio")
plt.title(f"O2 grid convergence for 1D WE in R{d}")

plt.legend()
plt.show()