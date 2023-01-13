import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym

def inbetweenAB(a, b, x):
    if x > a and x < b:
        return True
    else:
        return False


rmin, rmax = 0.1, 1.8
c = 1.5
a, b = 0.6, 1.2
d = 2

N = 600
h = (rmax - rmin) / N
courant = 0.5
tau = courant * h / c
TIME = 1.25
T = int(TIME / tau)
print(T)

# t0 = (0.9 + 0.3) / c
t0 = ((b + a) / 2 + 0.5) / c

r_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)

r = sym.symbols("r")

with open(f"3-d_task/d{d}_N{N}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U_answer = np.loadtxt(file)

U_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
U_fun = sym.lambdify(r, U_fun_expr, "numpy")

fig, ax = plt.subplots(2, 1)
fig.tight_layout(pad=2)

ymin = -0.1
ymax = np.max(U_answer) + 0.05

for oneax in ax:
    oneax.set_ylim((ymin, ymax))

line1, = ax[0].plot([], [], color="r")
line2, = ax[1].plot([], [], color="b")
line = [line1, line2]

def animate_comp(n):
    ax[0].clear()
    ax[0].set_title("Computational solution")

    ax[1].clear()
    ax[1].set_title("Analitycal solution")

    for oneax in ax:
        oneax.set_ylim((ymin, ymax))

    U_out = np.zeros(N + 2)
    for i in range(N + 2):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_out[i] = U_fun(c*(n*tau + t0) - r_grid[i])
    
    line[0] = ax[0].plot(r_grid, U_answer[n], "r")
    line[1] = ax[1].plot(r_grid, r_grid**((1-d)/2) * U_out, "b")
    if n % 50 == 0: print(n)
    return line

def animate_analit(n):
    ax[1].clear()
    ax[1].set_title("Analitycal solution")
    U_out = np.zeros(N + 2)
    for i in range(N + 2):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_out[i] = U_fun(c*(n*tau + t0) - r_grid[i])
    return ax[1].plot(r_grid, r_grid**((1-d)/2) * U_out, "b")

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

# computational_sol.save(f"3-d_task/combined_d{d}.gif", writer=animation.PillowWriter(fps=24))
# analitycal_sol.save("3-d_task/analitycal.gif", writer=animation.PillowWriter(fps=24))

plt.show()