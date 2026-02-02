from goph547lab01.gravity import gravity_effect_point, gravity_potential_point
import numpy as np
import matplotlib.pyplot as plt

m = 1e7
xm = np.array([0, 0, -10])
z_levels = [0, 10, 100]

x_25, y_25 = np.meshgrid(
    np.linspace(-100, 100, 9), np.linspace(-100,100, 9)
)

x_5, y_5 = np.meshgrid(
    np.linspace(-100, 100, 41), np.linspace(-100,100, 41)
)

def compute_field(x_5, y_5, x_25, y_25, z_levels, xm, m):

    U_5  = np.zeros((x_5.shape[0],  x_5.shape[1],  len(z_levels)))
    gz_5 = np.zeros((x_5.shape[0],  x_5.shape[1],  len(z_levels)))

    U_25  = np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels)))
    gz_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels)))

    # ---- 5 m grid ----
    for k, z in enumerate(z_levels):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x = np.array([x_5[i, j], y_5[i, j], z])
                U_5[i, j, k]  = gravity_potential_point(x, xm, m)
                gz_5[i, j, k] = gravity_effect_point(x, xm, m)

    # ---- 25 m grid ----
    for k, z in enumerate(z_levels):
        for i in range(x_25.shape[0]):
            for j in range(x_25.shape[1]):
                x = np.array([x_25[i, j], y_25[i, j], z])
                U_25[i, j, k]  = gravity_potential_point(x, xm, m)
                gz_25[i, j, k] = gravity_effect_point(x, xm, m)

    return U_5, gz_5, U_25, gz_25

U_5, gz_5, U_25, gz_25 = compute_field(
    x_5, y_5, x_25, y_25, z_levels, xm, m
)

Umin = min(np.min(U_5), np.min(U_25))
Umax = max(np.max(U_5), np.max(U_25))

gzmin = min(np.min(gz_5), np.min(gz_25))
gzmax = max(np.max(gz_5), np.max(gz_25))



fig, axes = plt.subplots(3, 2, figsize=(12, 16))

for k, z in enumerate(z_levels):

    # ---- Potential ----
    axU = axes[k, 0]
    cU = axU.contourf(
        x_5, y_5, U_5[:, :, k],
        levels=20, vmin=Umin, vmax=Umax, cmap="viridis"
    )
    axU.plot(x_5, y_5, "xk", markersize=2)
    fig.colorbar(cU, ax=axU)
    axU.set_title(f"U at z = {z} m")
    axU.set_xlabel("x (m)")
    axU.set_ylabel("y (m)")
    axU.set_aspect("equal")

    # ---- Gravity ----
    axG = axes[k, 1]
    cG = axG.contourf(
        x_5, y_5, gz_5[:, :, k],
        levels=20, vmin=gzmin, vmax=gzmax, cmap="plasma"
    )
    axG.plot(x_5, y_5, "xk", markersize=2)
    fig.colorbar(cG, ax=axG)
    axG.set_title(f"gₙ at z = {z} m")
    axG.set_xlabel("x (m)")
    axG.set_ylabel("y (m)")
    axG.set_aspect("equal")

fig.suptitle("Point Mass Gravity Fields – (dx = 5 m)", fontsize=20)
plt.savefig('../figures/Point Mass Gravity Fields (dx = 5 m).png')

fig, axes = plt.subplots(3, 2, figsize=(12, 16))

for k, z in enumerate(z_levels):

    # ---- Potential ----
    axU = axes[k, 0]
    cU = axU.contourf(
        x_25, y_25, U_25[:, :, k],
        levels=20, vmin=Umin, vmax=Umax, cmap="viridis"
    )
    axU.plot(x_25, y_25, "xk", markersize=2)
    fig.colorbar(cU, ax=axU)
    axU.set_title(f"U at z = {z} m")
    axU.set_xlabel("x (m)")
    axU.set_ylabel("y (m)")
    axU.set_aspect("equal")

    # ---- Gravity ----
    axG = axes[k, 1]
    cG = axG.contourf(
        x_25, y_25, gz_25[:, :, k],
        levels=20, vmin=gzmin, vmax=gzmax, cmap="plasma"
    )
    axG.plot(x_25, y_25, "xk", markersize=2)
    fig.colorbar(cG, ax=axG)
    axG.set_title(f"gₙ at z = {z} m")
    axG.set_xlabel("x (m)")
    axG.set_ylabel("y (m)")
    axG.set_aspect("equal")

fig.suptitle("Point Mass Gravity Fields – (dx = 25 m)", fontsize=20)
plt.savefig('../figures/Point Mass Gravity Fields (dx=25m).png')
