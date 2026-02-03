from examples.generate_mass_sets import generate_mass_set
from goph547lab01.gravity import gravity_effect_point
from goph547lab01.gravity import gravity_potential_point
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

m = 1e7
z_levels = [0, 10, 100]

x_25, y_25 = np.meshgrid(
    np.linspace(-100, 100, 9), np.linspace(-100, 100, 9)
)

x_5, y_5 = np.meshgrid(
    np.linspace(-100, 100, 41), np.linspace(-100, 100, 41)
)

def calc_U_and_gz(U, gz, x, y, mass_set, xm):
    for k, z in enumerate(z_levels):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_array = np.array([x[i, j], y[i, j], z])
                for mi, xmi in zip(mass_set, xm):
                    U[i, j, k]  += gravity_potential_point(x_array, xmi, mi)
                    gz[i, j, k] += gravity_effect_point(x_array, xmi, mi)
    return U, gz

def generate_plot(U, gz, x, y, Umin, Umax, gzmin, gzmax, grid, mass_set):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))

    for k, z in enumerate(z_levels):
        axU = axes[k, 0]
        cU = axU.contourf(
            x, y, U[:, :, k],
            levels=20, vmin=Umin, vmax=Umax, cmap="viridis"
        )
        axU.plot(x, y, "xk", markersize=2)
        fig.colorbar(cU, ax=axU)
        axU.set_title(f"U at z = {z}m")
        axU.set_xlabel("x (m)")
        axU.set_ylabel("y (m)")
        axU.set_aspect("equal")

        axG = axes[k, 1]
        cG = axG.contourf(
            x, y, gz[:, :, k],
            levels=20, vmin=gzmin, vmax=gzmax, cmap="plasma"
        )
        axG.plot(x, y, "xk", markersize=2)
        fig.colorbar(cG, ax=axG)
        axG.set_title(f"gz at z = {z}m")
        axG.set_xlabel("x (m)")
        axG.set_ylabel("y (m)")
        axG.set_aspect("equal")

    fig.suptitle(f"Multi Mass Gravity Fields – (dx = {grid}m) - mass set {mass_set}", fontsize=20)
    plt.savefig(f'../figures/Multi Mass Gravity Fields – (dx = {grid}m) - mass set {mass_set}.png')

def main():
    d1 = loadmat("mass_set_1.mat")
    d2 = loadmat("mass_set_2.mat")
    d3 = loadmat("mass_set_3.mat")

    mass_set_1 = d1["mass_set_1"].squeeze()
    mass_set_2 = d2["mass_set_2"].squeeze()
    mass_set_3 = d3["mass_set_3"].squeeze()
    xm = generate_mass_set()[1]

    U_5_m1, gz_5_m1 = calc_U_and_gz(
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        x_5, y_5, mass_set_1, xm
    )

    U_5_m2, gz_5_m2 = calc_U_and_gz(
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        x_5, y_5, mass_set_2, xm
    )

    U_5_m3, gz_5_m3 = calc_U_and_gz(
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels))),
        x_5, y_5, mass_set_3, xm
    )

    U_25_m1, gz_25_m1 = calc_U_and_gz(
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        x_25, y_25, mass_set_1, xm
    )

    U_25_m2, gz_25_m2 = calc_U_and_gz(
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        x_25, y_25, mass_set_2, xm
    )

    U_25_m3, gz_25_m3 = calc_U_and_gz(
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        np.zeros((x_25.shape[0], x_25.shape[1], len(z_levels))),
        x_25, y_25, mass_set_3, xm
    )

    Umin_m1 = min(np.min(U_5_m1), np.min(U_25_m1))
    Umax_m1 = max(np.max(U_5_m1), np.max(U_25_m1))
    gzmin_m1 = min(np.min(gz_5_m1), np.min(gz_25_m1))
    gzmax_m1 = max(np.max(gz_5_m1), np.max(gz_25_m1))

    Umin_m2 = min(np.min(U_5_m2), np.min(U_25_m2))
    Umax_m2 = max(np.max(U_5_m2), np.max(U_25_m2))
    gzmin_m2 = min(np.min(gz_5_m2), np.min(gz_25_m2))
    gzmax_m2 = max(np.max(gz_5_m2), np.max(gz_25_m2))

    Umin_m3 = min(np.min(U_5_m3), np.min(U_25_m3))
    Umax_m3 = max(np.max(U_5_m3), np.max(U_25_m3))
    gzmin_m3 = min(np.min(gz_5_m3), np.min(gz_25_m3))
    gzmax_m3 = max(np.max(gz_5_m3), np.max(gz_25_m3))

    generate_plot(U_5_m1, gz_5_m1, x_5, y_5, Umin_m1, Umax_m1, gzmin_m1, gzmax_m1, 5, 1)
    generate_plot(U_25_m1, gz_25_m1, x_25, y_25, Umin_m1, Umax_m1, gzmin_m1, gzmax_m1, 25, 1)

    generate_plot(U_5_m2,gz_5_m2, x_5, y_5, Umin_m2, Umax_m2, gzmin_m2, gzmax_m2, 5, 2)
    generate_plot(U_25_m2, gz_25_m2, x_25, y_25, Umin_m2, Umax_m2, gzmin_m2, gzmax_m2, 25, 2)

    generate_plot(U_5_m3, gz_5_m3, x_5, y_5, Umin_m3, Umax_m3, gzmin_m3, gzmax_m3, 5, 3)
    generate_plot(U_25_m3, gz_25_m3, x_25, y_25, Umin_m3, Umax_m3, gzmin_m3, gzmax_m3, 25, 3)
if __name__ == "__main__":
    main()
