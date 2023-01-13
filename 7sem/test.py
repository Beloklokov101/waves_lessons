import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import matplotlib.animation as animation

rmin, rmax = 0.1, 1.8
c = 1.5
a, b = 0.6, 1.2
approxLevel = 4
approxLevel_2 = int(approxLevel / 2)
d = 1

NStart = 200
h = (rmax - rmin) / NStart
courant = 0.5
tau = courant * h / c #поправочный коэффициент 
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