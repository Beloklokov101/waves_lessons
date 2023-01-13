import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym

def inbetweenAB(a, b, x):
    if x > a and x < b:
        return True
    else:
        return False

d = 2
N = 200
TIME = 1.25
courant = 0.5

rmin, rmax = 0.1, 1.8
c = 1.5
h = (rmax - rmin) / N
tau = courant * h / c
T = int(TIME / tau)
print(T, "\n")
r_grid = np.linspace(rmin - h / 2, rmax + h / 2, N + 2)

with open(f"3-d_task/d{d}_N{N}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U200 = np.loadtxt(file)

with open(f"3-d_task/d{d}_N{N * 3}_T{TIME}_c{c}_courant{courant}.txt", "r") as file:
    U600 = np.loadtxt(file)

r = sym.symbols("r")

a, b = 0.6, 1.2
t0 = ((b + a) / 2 + 0.5) / c
U_fun_expr = sym.exp(-4 * (2*r - (a + b))**2 / ((b - a)**2 - (2*r - (a + b))**2))
U_fun = sym.lambdify(r, U_fun_expr, "numpy")


# fig = plt.figure()
# ax = plt.subplot()
fig, ax = plt.subplots(2, 1)
# ax.set_ylim((0, 1))

line1, = ax[0].plot([], [], color="r")
line2, = ax[1].plot([], [], color="b")
line = [line1, line2]

def animate(n):
    ax[0].clear()
    ax[1].clear()
    # ax.set_ylim((0, 1))
    if n % 50 == 0: print(n)

    # conv_C = np.max(np.abs(U200[n, 1:-1] - U_analit[1:-1])) / \
    #     np.max(np.abs(U600[3*n, 2::3] - U_analit[1:-1]))

    # conv_L2 = np.sqrt(np.sum((U200[n, 1:-1] - U_analit[1:-1])**2)) / \
    #     np.sqrt(np.sum((U600[3*n, 2::3] - U_analit[1:-1])**2))

    U_analit = np.zeros(N + 2)
    for i in range(N + 2):
        if inbetweenAB(a, b, c*(n*tau + t0) - r_grid[i]):
            U_analit[i] = U_fun(c*(n*tau + t0) - r_grid[i])

    line[0] = ax[0].plot(r_grid[1:-1], U200[n, 1:-1] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1])
    line[1] = ax[1].plot(r_grid[1:-1], U600[3*n, 2::3] - r_grid[1:-1]**((1-d)/2) * U_analit[1:-1])

    return line

interval_animation = 10
repeat_animation = True
firstAnim = animation.FuncAnimation(fig, 
                                    animate, 
                                    np.arange(T + 1),
                                    # 1100,
                                    interval = interval_animation,
                                    repeat = repeat_animation)

# firstAnim.save(f"1-st_task/d{d}.gif", writer=animation.PillowWriter(fps=24))

plt.show()