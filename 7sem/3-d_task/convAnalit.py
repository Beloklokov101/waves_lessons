import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def inbetweenAB(a, b, x):
    if x > a and x < b:
        return True
    else:
        return False

d = 2
NStart = 200
TIME = 1.25
courant = 0.5

rmin, rmax = 0.1, 1.8
c = 1.5
h = (rmax - rmin) / NStart
tau = courant * h / c
T = int(TIME / tau)
print(T)
# x_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)

with open(f"3-d_task/d{d}_N{NStart}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U200 = np.loadtxt(file)

with open(f"3-d_task/d{d}_N{NStart * 3}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U600 = np.loadtxt(file)

print(f"U200 shape = {np.shape(U200[:, 1:-1])}")
print(f"U600 shape = {np.shape(U600[:-1:3, 2::3])}")


r = sym.symbols("r")

a, b = 0.6, 1.2
t0 = ((b + a) / 2 + 0.5) / c
U_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
U_fun = sym.lambdify(r, U_fun_expr, "numpy")

r_grid = np.linspace(rmin - h / 2, rmax + h / 2, NStart + 2)

T = min(np.shape(U200[:, 1:-1])[0], np.shape(U600[:-1:3, 2::3])[0])
# print(np.shape(U200[:, 1:-1])[0], np.shape(U600[:-1:3, 2::3])[0])

conv_C = np.zeros(T + 1)
conv_L2 = np.zeros(T + 1)
for n in range(T + 1):
    U_analit = np.zeros(NStart + 2)
    for i in range(NStart + 2):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_analit[i] = U_fun(c*(n*tau + t0) - r_grid[i])
    
    conv_C[n] = np.max(np.abs(U200[n, 1:-1] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1])) / \
        np.max(np.abs(U600[3*n, 2::3] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1]))
    # removed h 
    conv_L2[n] = np.sqrt(np.sum((U200[n, 1:-1] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1])**2)) / \
        np.sqrt(np.sum((U600[3*n, 2::3] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1])**2))

# with open("3-d_task/conv_C.txt", "w") as file:
#     np.savetxt(file, conv_C)

# with open("3-d_task/conv_L2.txt", "w") as file:
#     np.savetxt(file, conv_L2)


fig = plt.figure()
# fig.set_label(f"O2 grid convergence for 1D WE in R{d}")
ax = plt.subplot()
time_grid = np.linspace(0, TIME, T + 1)
ax.plot(time_grid, conv_C, "g", label="C-norm")
ax.plot(time_grid, conv_L2, "b", label="L2-norm")
ax.plot(time_grid, 9*np.ones(T + 1), "r", label="Ref=9")
ax.set_xlabel("Time")
ax.set_ylabel("Ratio")
plt.title(f"O2 grid convergence for 1D WE in R{d}")

plt.legend()
plt.show()