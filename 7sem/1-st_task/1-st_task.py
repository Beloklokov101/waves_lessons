import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import matplotlib.animation as animation

rmin, rmax = 0, 1.8
c = 1.5
a, b = 0.6, 1.2
d = 1

N = 200
h = (rmax - rmin) / N
courant = 0.5
tau = courant * h / c
TIME = 1.5
T = int(TIME / tau)
print(f"T = {T}\n")

x_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)


r = sym.symbols("r")

cond_ab = (a <= x_grid) & (x_grid <= b)

v0_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
v0_fun = sym.lambdify(r, v0_fun_expr, "numpy")
v0 = np.zeros(N + 2)
v0[cond_ab] = v0_fun(x_grid[cond_ab])
"""
v0_der_expr = sym.diff(v0_fun_expr, r)
v0_der_fun = sym.lambdify(r, v0_der_expr, "numpy")
v0_der = np.zeros(N + 1)
v0_der[cond_ab] = v0_der_fun(x_grid[cond_ab])

v0_2der_expr = sym.diff(v0_der_expr, r)
v0_2der_fun = sym.lambdify(r, v0_2der_expr, "numpy")
v0_2der = np.zeros(N + 1)
v0_2der[cond_ab] = v0_2der_fun(x_grid[cond_ab])
"""
d2u_v0_expr = sym.diff(r**(d - 1) * sym.diff(v0_fun_expr, r), r) / r**(d - 1)
d2u_v0_fun = sym.lambdify(r, d2u_v0_expr, "numpy")
d2u_v0 = np.zeros(N + 2)
d2u_v0[cond_ab] = d2u_v0_fun(x_grid[cond_ab])

u_n = v0 + tau**2 / 2 * c**2 * d2u_v0
u_prev = v0.copy()
u_next = np.zeros(N + 2)

U_answer = np.zeros((T + 1, N + 2))
U_answer[0] = u_prev.copy()
U_answer[1] = u_n.copy()

for n in range(2, T + 1):
    for i in range(1, N + 1):
        u_next[i] = 2*u_n[i] - u_prev[i] + (tau*c)**2 / (h*x_grid[i]**(d - 1)) * \
        (((x_grid[i] + x_grid[i + 1]) / 2)**(d - 1) * (u_n[i + 1] - u_n[i]) / h - \
        ((x_grid[i] + x_grid[i - 1]) / 2)**(d - 1) * (u_n[i] - u_n[i - 1]) / h)
    u_next[0] = u_next[1]
    u_next[N + 1] = u_next[N]

    u_prev = u_n.copy()
    u_n = u_next.copy()
    u_next = np.zeros(N + 2)
    U_answer[n] = u_n.copy()

    if n % 50 == 0: print(f"n = {n}")


# with open(f"d{d}_N{N}_T{TIME}_c{c}_courant{courant}.txt", "x") as file:
#     np.savetxt(file, U_answer)

fig = plt.figure()
ax = plt.subplot()

def animate(i):
    ax.clear()
    return ax.plot(x_grid, U_answer[i])

interval_animation = 10
repeat_animation = True
corner_animation = animation.FuncAnimation(fig, 
                                      animate, 
                                      np.arange(T + 1),
                                      interval = interval_animation,
                                      repeat = repeat_animation)

plt.show()