import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def inbetweenAB(a, b, x):
    if x > a and x < b:
        return True
    else:
        return False

# rmin, rmax = 0, 1.8
rmin, rmax = 0.1, 1.8
c = 1.5
a, b = 0.6, 1.2
approxLevel = 4
approxLevel_2 = int(approxLevel / 2)
d = 1

NStart = 200
h = (rmax - rmin) / NStart
courant = 0.5
tau_mult = 1
tau = tau_mult * courant * h / c #поправочный коэффициент 
TIME = 0.5
T = int(TIME / tau)
print(f"T = {T}\n")

r_grid = np.linspace(rmin - h / 2 * (approxLevel - 1), rmax + h / 2 * (approxLevel - 1), NStart + approxLevel)

r = sym.symbols("r")

t0 = ((b + a) / 2 + 0.5) / c

U_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
U_fun = sym.lambdify(r, U_fun_expr, "numpy")

with open(f"5-th_task/N{NStart}_1tau.txt", "r") as file:
    U200 = np.loadtxt(file)

with open(f"5-th_task/N{NStart}_0.5tau.txt", "r") as file:
    U200_2 = np.loadtxt(file)

with open(f"5-th_task/N{3 * NStart}_1tau.txt", "r") as file:
    U600 = np.loadtxt(file)

with open(f"5-th_task/N{3 * NStart}_0.5tau.txt", "r") as file:
    U600_2 = np.loadtxt(file)

# print(f"Utau shape = {np.shape(Utau)}")
# print(f"Utau2 shape = {np.shape(Utau2)}")

print(f"Utau shape = {np.shape(U200[:, approxLevel_2 : -approxLevel_2])}")
print(f"Utau2 shape = {np.shape(U200_2[::2, approxLevel_2 : -approxLevel_2])}")
print(f"Utau4 shape = {np.shape(U600[::3, approxLevel_2 + 1 : - approxLevel_2 - 1 : 3])}")
print(f"Utau4 shape = {np.shape(U600_2[::6, approxLevel_2 + 1 : - approxLevel_2 - 1 : 3])}")

U200sliced = U200[:, approxLevel_2 : -approxLevel_2]
U200_2sliced = U200_2[::2, approxLevel_2 : -approxLevel_2]
U600sliced = U600[::3, approxLevel_2 + 1 : - approxLevel_2 - 1 : 3]
U600_2sliced = U600_2[::6, approxLevel_2 + 1 : - approxLevel_2 - 1 : 3]

W200 = (4 * U200_2sliced - U200sliced) / 3
W600 = (4 * U600_2sliced - U600sliced) / 3

# T = min(np.shape(Utausliced)[0], np.shape(Utau2sliced)[0])
# print(np.shape(Utau[:, 1:-1])[0], np.shape(Utau2[:-1:3, 2::3])[0])

conv_C = np.zeros(T + 1)
conv_L2 = np.zeros(T + 1)
for n in range(T + 1):
    U_analit = np.zeros(NStart + approxLevel)
    for i in range(NStart + approxLevel):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_analit[i] = U_fun(c*(n*tau + t0) - r_grid[i])
    U_analit_sliced = U_analit[approxLevel_2 : -approxLevel_2]
    
    # d = 1
    conv_C[n] = np.max(np.abs(W200[n] - U_analit_sliced)) / \
        np.max(np.abs(W600[n] - U_analit_sliced))
    # removed h 
    conv_L2[n] = np.sqrt(np.sum((W200[n] - U_analit_sliced)**2)) / \
        np.sqrt(np.sum((W600[n] - U_analit_sliced)**2))

# with open(f"5-th_task/approx{approxLevel}_conv_C.txt", "w") as file:
#     np.savetxt(file, conv_C)

# with open(f"5-th_task/approx{approxLevel}_conv_L2.txt", "w") as file:
#     np.savetxt(file, conv_L2)


fig = plt.figure()
# fig.set_label(f"O2 grid convergence for 1D WE in R{d}")
ax = plt.subplot()
time_grid = np.linspace(0, TIME, T + 1)
ax.plot(time_grid, conv_C, "g", label="C-norm")
ax.plot(time_grid, conv_L2, "b", label="L2-norm")
ax.plot(time_grid, 3**(approxLevel) * np.ones(T + 1), "r", label=f"Ref={3**(approxLevel)}")
ax.set_xlabel("Time")
ax.set_ylabel("Ratio")
plt.title("O44 grid convergence for 1D WE in R1")

plt.legend()
plt.show()