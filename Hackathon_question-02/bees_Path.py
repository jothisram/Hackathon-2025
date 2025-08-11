# Requirements:
# pip install numpy scipy matplotlib

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# Parameters
a = 10.0
b = 28.0
c = 2.667

# Initial conditions
x0, y0, z0 = 0.0, 1.0, 1.05
y_init = np.array([x0, y0, z0])

# Time span and sampling
t0, t1 = 0.0, 50.0     # simulate for 50 time units
num_points = 20000      # dense sampling for smooth curve
t_eval = np.linspace(t0, t1, num_points)

# Dynamical system
def f(t, Y):
    x, y, z = Y
    dxdt = a * (y - b)
    dydt = b * x - y - x * z
    dzdt = x * y - c * z
    return [dxdt, dydt, dzdt]

# Solve ODE
sol = solve_ivp(
    f, (t0, t1), y_init,
    t_eval=t_eval,
    method="RK45",          # good general-purpose solver
    rtol=1e-8, atol=1e-10   # tight tolerances for accuracy
)

if not sol.success:
    raise RuntimeError(f"ODE solver failed: {sol.message}")

x, y, z = sol.y

# Plotting
plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
# Color by time for progression cue
color_vals = np.linspace(0, 1, len(sol.t))
ax.plot3D(x, y, z, lw=0.7, color='tab:blue')
sc = ax.scatter(x[::500], y[::500], z[::500], c=color_vals[::500], cmap='viridis', s=5)

ax.set_title("3D Trajectory of the Dynamical System (Bee's Path)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=25, azim=135)
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label("Normalized time")
plt.tight_layout()
plt.show()
