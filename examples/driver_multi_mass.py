from goph547lab01.gravity import gravity_effect_point
from goph547lab01.gravity import gravity_potential_point
from examples.generate_mass_sets import generate_mass_set, mass_set_2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

load_m1 = loadmat("mass_set_1.mat")
mass_set_1 = load_m1['mass_set_1'].squeeze()
load_m2 = loadmat("mass_set_2.mat")
mass_set_2 = load_m2['mass_set_2'].squeeze()
load_m3 = loadmat("mass_set_3.mat")
mass_set_3 = load_m3['mass_set_3'].squeeze()
xm = generate_mass_set()[1]

m = 1e7
z_levels = [0, 10, 100]

x_25, y_25 = np.meshgrid(
    np.linspace(-100, 100, 9), np.linspace(-100,100, 9)
)

x_5, y_5 = np.meshgrid(
    np.linspace(-100, 100, 41), np.linspace(-100,100, 41)
)



U_5_zeros  = np.zeros((x_5.shape[0],  x_5.shape[1],  len(z_levels)))
gz_5_zeros = np.zeros((x_5.shape[0],  x_5.shape[1],  len(z_levels)))

U_25_zeros  = np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels)))
gz_25_zeros = np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels)))

# ---- 5 m grid ----
def calc_U_and_gz(U, gz, x, y, mass_set):
    for k, z in enumerate(z_levels):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_array = np.array([x[i, j], y[i, j], z])
                for mi, xmi in zip(mass_set, xm):
                    U[i, j, k]  += gravity_potential_point(x_array, xmi, mi)
                    gz[i, j, k] += gravity_effect_point(x_array, xmi, mi)

    return U, gz

U_5, gz_5 = calc_U_and_gz(U_5_zeros, gz_5_zeros, x_25, y_25, mass_set_1)
U_5 , gz_25 =


Umin = min(np.min(U_5), np.min(U_25))
Umax = max(np.max(U_5), np.max(U_25))

gzmin = min(np.min(gz_5), np.min(gz_25))
gzmax = max(np.max(gz_5), np.max(gz_25))

def generate_plot(U, gz, x, y, grid):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))

    for k, z in enumerate(z_levels):

        # ---- Potential ----
        axU = axes[k, 0]
        cU = axU.contourf(
            x, y, U[:, :, k],
            levels=20, vmin=Umin, vmax=Umax, cmap="viridis"
        )
        axU.plot(x_5, y_5, "xk", markersize=2)
        fig.colorbar(cU, ax=axU)
        axU.set_title(f"U at z = {z}m")
        axU.set_xlabel("x (m)")
        axU.set_ylabel("y (m)")
        axU.set_aspect("equal")

        # ---- Gravity ----
        axG = axes[k, 1]
        cG = axG.contourf(
            x, y, gz[:, :, k],
            levels=20, vmin=gzmin, vmax=gzmax, cmap="plasma"
        )
        axG.plot(x_5, y_5, "xk", markersize=2)
        fig.colorbar(cG, ax=axG)
        axG.set_title(f"gz at z = {z}m")
        axG.set_xlabel("x (m)")
        axG.set_ylabel("y (m)")
        axG.set_aspect("equal")

    fig.suptitle(f"Multi Mass Gravity Fields – (dx = {grid}m)", fontsize=20)
    plt.show()

generate_plot(U_5, gz_5, x_5, y_5, 5)

