import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import matplotlib.animation as animation

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

N = 600
h = (rmax - rmin) / N
courant = 0.5
tau_mult = 0.5
tau = tau_mult * courant * h / c #поправочный коэффициент 
TIME = 0.5
T = int(TIME / tau)
print(f"T = {T}\n")

r_grid = np.linspace(rmin - h / 2 * (approxLevel - 1), rmax + h / 2 * (approxLevel - 1), N + approxLevel)

r = sym.symbols("r")

# cond_ab = (a <= r_grid) & (r_grid <= b)
# t0 = (0.9 + 0.3) / c
t0 = ((b + a) / 2 + 0.5) / c

U_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
U_fun = sym.lambdify(r, U_fun_expr, "numpy")
# v0 = np.zeros(N + 2)
# v0[cond_ab] = v0_fun(r_grid[cond_ab])

U_der_expr = sym.diff(U_fun_expr, r)
U_der_fun = sym.lambdify(r, U_der_expr, "numpy")
# v0_der = np.zeros(N + 2)
# v0_der[cond_ab] = v0_der_fun(r_grid[cond_ab])

u_prev = np.zeros(N + approxLevel)
for i in range(N + approxLevel):
    if inbetweenAB(a, b, c*t0 - r_grid[i]):
        u_prev[i] = r_grid[i]**((1-d)/2) * U_fun(c*t0 - r_grid[i])
# u_prev = r_grid**((1-d)/2) * U_fun(c*t0 - r_grid)

u_n = np.zeros(N + approxLevel)
for i in range(N + approxLevel):
    if inbetweenAB(a, b, c*(tau + t0) - r_grid[i]):
        u_n[i] = r_grid[i]**((1-d)/2) * U_fun(c*(tau + t0) - r_grid[i])
# u_n = r_grid**((1-d)/2) * U_fun(c*(tau + t0) - r_grid)

u_next = np.zeros(N + approxLevel)

U_answer = np.zeros((T + 1, N + approxLevel))
U_answer[0] = u_prev.copy()
U_answer[1] = u_n.copy()

if (approxLevel == 4):
    a_taylor = [-5/4, 4/3, -1/12]
elif (approxLevel == 6):
    a_taylor = [-49/36, 3/2, -3/20, 1/90]

for n in range(1, T):
    for i in range(approxLevel_2, N + approxLevel_2):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_in = U_fun(c*(n*tau + t0) - r_grid[i])
        else:
            U_in = 0

        taylor_sum = 0
        for k in range(approxLevel_2 + 1):
            taylor_sum += a_taylor[k] * (u_n[i - k] + u_n[i + k])

        u_next[i] = 2*u_n[i] - u_prev[i] + (tau*c/h)**2 * taylor_sum 
    
    for j in range(approxLevel_2):
        u_next[j] = u_next[approxLevel - 1 - j]
        u_next[N + approxLevel - 1 - j] = u_next[N + j]

    u_prev = u_n.copy()
    u_n = u_next.copy()
    u_next = np.zeros(N + approxLevel)
    U_answer[n + 1] = u_n.copy()

    if n % 50 == 0: print(f"n = {n}\n")


with open(f"5-th_task/N{N}_{tau_mult}tau.txt", "w") as file:
    np.savetxt(file, U_answer)


# fig, ax = plt.subplots(2, 1)
fig = plt.figure()
ax = plt.subplot()

def animate_comp(n):
    ax.clear()
    ax.set_title("Computational solution")
    return ax.plot(r_grid, U_answer[n], "r")

# def animate_analit(n):
#     ax[1].clear()
#     ax[1].set_title("Analitycal solution")
#     U_out = np.zeros(N + 2)
#     for i in range(N + 2):
#         if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
#             U_out[i] = U_fun(c*(n*tau + t0) - r_grid[i])
#     return ax[1].plot(r_grid, r_grid**((1-d)/2) * U_out, "b")

interval_animation = 10
repeat_animation = True
computational_sol = animation.FuncAnimation(fig, 
                                      animate_comp, 
                                      np.arange(T + 1),
                                      interval = interval_animation,
                                      repeat = repeat_animation)

# analitycal_sol = animation.FuncAnimation(fig, 
#                                       animate_analit, 
#                                       np.arange(T + 1),
#                                       interval = interval_animation,
#                                       repeat = repeat_animation)

plt.show()