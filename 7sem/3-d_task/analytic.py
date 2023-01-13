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
d = 3

N = 600
h = (rmax - rmin) / N
courant = 0.5
tau = courant * h / c
TIME = 1.25
T = int(TIME / tau)
print(f"T = {T}\n")

r_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)

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

u_prev = np.zeros(N + 2)
for i in range(N + 2):
    if inbetweenAB(a, b, c*t0 - r_grid[i]):
        u_prev[i] = r_grid[i]**((1-d)/2) * U_fun(c*t0 - r_grid[i])
# u_prev = r_grid**((1-d)/2) * U_fun(c*t0 - r_grid)

u_n = np.zeros(N + 2)
for i in range(N + 2):
    if inbetweenAB(a, b, c*(tau + t0) - r_grid[i]):
        u_n[i] = r_grid[i]**((1-d)/2) * U_fun(c*(tau + t0) - r_grid[i])
# u_n = r_grid**((1-d)/2) * U_fun(c*(tau + t0) - r_grid)

u_next = np.zeros(N + 2)

U_answer = np.zeros((T + 1, N + 2))
U_answer[0] = u_prev.copy()
U_answer[1] = u_n.copy()

for n in range(1, T):
    for i in range(1, N + 1):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_in = U_fun(c*(n*tau + t0) - r_grid[i])
        else:
            U_in = 0

        u_next[i] = 2*u_n[i] - u_prev[i] + (tau*c)**2 / (h*r_grid[i]**(d - 1)) * \
        (((r_grid[i] + r_grid[i + 1]) / 2)**(d - 1) * (u_n[i + 1] - u_n[i]) / h - \
        ((r_grid[i] + r_grid[i - 1]) / 2)**(d - 1) * (u_n[i] - u_n[i - 1]) / h) + \
        tau**2 * c**2 * (d-1)*(d-3) * U_in / (4 * r_grid[i]**((d+3)/2))
    
    if inbetweenAB(a, b, c*(n*tau + t0) - rmin):
        u_next[0] = u_next[1] - h * ((1-d)/2 * rmin**(-(1+d)/2) * U_fun(c*(n*tau + t0) - rmin) - \
        rmin**((1-d)/2) * U_der_fun(c*(n*tau + t0) - rmin))
    else:
        u_next[0] = u_next[1]

    if inbetweenAB(a, b, c*(n*tau + t0) - rmax):
        u_next[N + 1] = u_next[N] + h * ((1-d)/2 * rmax**(-(1+d)/2) * U_fun(c*(n*tau + t0) - rmax) - \
        rmax**((1-d)/2) * U_der_fun(c*(n*tau + t0) - rmax))
    else:
        u_next[N + 1] = u_next[N]

    u_prev = u_n.copy()
    u_n = u_next.copy()
    u_next = np.zeros(N + 2)
    U_answer[n + 1] = u_n.copy()

    if n % 50 == 0: print(f"n = {n}\n")


with open(f"3-d_task/d{d}_N{N}_T{TIME}_c{c}_courant{courant}.txt", "w") as file:
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