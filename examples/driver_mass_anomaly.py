import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIG_DIR, exist_ok=True)


from scipy.io import loadmat
from goph547lab01.gravity import gravity_effect_point
import matplotlib.pyplot as plt
import numpy as np

data = loadmat('anomaly_data.mat')

x = data['x']
y = data['y']
z = data['z']
rho = data['rho']

cell_vol = 2 ** 3

mass_cells = cell_vol * rho
total_mass = np.sum(mass_cells)

xa = np.sum(mass_cells * x) / total_mass
ya = np.sum(mass_cells * y) / total_mass
za = np.sum(mass_cells * z) / total_mass

anomalyPosition = [xa, ya, za]

print(f'Mass of anomaly: {total_mass}')
print(f'Anomaly coordinates: {anomalyPosition}')
print(f'Max cell density: {np.max(rho)}')
print(f'Mean cell density: {np.mean(rho)}')

x_vec = x[0, :, 0]
y_vec = y[:, 0, 0]
z_vec = z[0, 0, :]

rho_xz = np.mean(rho, axis=0)
X_xz, Z_xz = np.meshgrid(x_vec, z_vec)

rho_yz = np.mean(rho, axis=1)
Y_yz, Z_yz = np.meshgrid(y_vec, z_vec)

rho_xy = np.mean(rho, axis=2)
X_xy, Y_xy = np.meshgrid(x_vec, y_vec)


#Plot density cross-sections
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

c0 = axes[0].contourf(X_xz, Z_xz, rho_xz.T, levels=20)
axes[0].plot(xa, za, 'xk', markersize=3)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_xlim(-20,20)
axes[0].set_ylim(0,-20)
axes[0].set_title('Anomaly in xz-plane')
cbar0 = fig.colorbar(c0, ax=axes[0])

c1 = axes[1].contourf(Y_yz, Z_yz, rho_yz.T, levels=20)
axes[1].plot(ya, za, 'xk', markersize=3)
axes[1].set_xlabel('y (m)')
axes[1].set_ylabel('z (m)')
axes[1].set_xlim(-20,20)
axes[1].set_ylim(0,-20)
axes[1].set_title('Anomaly in yz-plane')
fig.colorbar(c0, ax=axes[1])

c2 = axes[2].contourf(X_xy, Y_xy, rho_xy.T, levels=20)
axes[2].plot(xa, ya, 'xk', markersize=3)
axes[2].set_xlabel('x (m)')
axes[2].set_ylabel('y (m)')
axes[2].set_xlim(-20,20)
axes[2].set_ylim(25,-25)
axes[2].set_title('Anomaly in xy-plane')
cbar2 = fig.colorbar(c0, ax=axes[2])


#Compute mean rho in cropped plot
x_xz_min, x_xz_max = -20, 20
z_xz_min, z_xz_max = -20, 0

y_yz_min, y_yz_max = x_xz_min, x_xz_max
z_yz_min, z_yz_max = z_xz_min, z_xz_max

x_xy_min, x_xy_max = x_xz_min, x_xz_max
y_xy_min, y_xy_max = -25,25



#Compute index for value cutoffs
ix_xz = np.where((x_vec >= x_xz_min) & (x_vec <= x_xz_max))[0]
iz_xz = np.where((z_vec >= z_xz_min) & (z_vec <= z_xz_max))[0]

iy_yz = np.where((y_vec >= y_yz_min) & (y_vec <= y_yz_max))[0]
iz_yz = np.where((z_vec >= z_yz_min) & (z_vec <= z_yz_max))[0]

ix_xy = np.where((x_vec >= x_xy_min) & (x_vec <= x_xy_max))[0]
iy_xy = np.where((y_vec >= y_xy_min) & (y_vec <= y_xy_max))[0]

rho_crop_xz = rho_xz[np.ix_(ix_xz, iz_xz)]
rho_crop_yz = rho_yz[np.ix_(iy_yz, iz_yz)]
rho_crop_xy = rho_xy[np.ix_(ix_xy, iy_xy)]

mean_rho_xz = np.mean(rho_crop_xz)
mean_rho_yz = np.mean(rho_crop_yz)
mean_rho_xy = np.mean(rho_crop_xy)

print(f"Mean rho in x-z plane: {mean_rho_xz}")
print(f"Mean rho in y-z plane: {mean_rho_yz}")
print(f"Mean rho in xy-plane: {mean_rho_xy}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Anomaly in 3 planes.png'))

z_levels = [0, 100]
mass = total_mass

x_5, y_5 = np.meshgrid(
    np.linspace(-100, 100, 41),
    np.linspace(-100, 100, 41)
)


#Initialize gravity array
gz = np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels)))

for k, z_obs in enumerate(z_levels):
    for i in range(x_5.shape[0]):
        for j in range(x_5.shape[1]):
            x = np.array([x_5[i, j], y_5[i, j], z_obs])
            gz[i, j, k] = gravity_effect_point(x, anomalyPosition, mass)

gz_min = np.min(gz)
gz_max = np.max(gz)


# Plot gravitational effect of anomaly data
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

for k, z_obs in enumerate(z_levels):
    ax = axes[k]
    c = ax.contourf(x_5, y_5, gz[:, :, k], levels=20, vmin=gz_min, vmax=gz_max, cmap = 'viridis_r')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Gravitational Effect at z = {z_obs} m')
    fig.colorbar(c, ax=ax)

fig.suptitle('Anomaly Gravity Effect at \nGround and Airborne Observation', fontsize=16)
plt.savefig(os.path.join(FIG_DIR, 'Anomaly Gravity Effect Forward Modelling.png'))


# Finite difference heights
z_levels_new = [1.0, 110.0]

# Initialize gravity array
gz_new = np.zeros((x_5.shape[0], x_5.shape[1], len(z_levels_new)))

# Forward model at new heights
for k, z_obs in enumerate(z_levels_new):
    for i in range(x_5.shape[0]):
        for j in range(x_5.shape[1]):
            obs_point = np.array([x_5[i, j], y_5[i, j], z_obs])
            gz_new[i, j, k] = gravity_effect_point(
                obs_point, anomalyPosition, mass
            )

#First Order Finite differences
dz0 = 1.0
dz10 = 10.0

dgdz_0 = (gz_new[:, :, 0] - gz[:, :, 0]) / dz0
dgdz_10 = (gz_new[:, :, 1] - gz[:, :, 1]) / dz10




#Second Order Finite Difference
dx = x_5[0,1] - x_5[0,0]
dy = y_5[1,0] - y_5[0,0]

def laplace(gz_slice,dx, dy):
    d2gzdx2 = np.gradient(np.gradient(gz_slice, dx, axis=1), dx, axis=1)
    d2gdy2 = np.gradient(np.gradient(gz_slice, dy, axis=0), dy, axis=0)
    return -(d2gzdx2 + d2gdy2)

d2gdz2_0_laplace = laplace(gz[:, :, 0], dx, dy)
d2gdz2_10_laplace = laplace(gz[:, :, 1], dx, dy)





#Plot First Order Finite Difference
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

ax = axes[0]
c = ax.contourf(x_5, y_5, dgdz_0, levels=20, cmap='viridis')
ax.set_title(r'$\partial g_z / \partial z$ at dz = 0 m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

ax = axes[1]
c = ax.contourf(x_5, y_5, dgdz_10, levels=20, cmap='viridis')
ax.set_title(r'$\partial g_z / \partial z$ at dz = 10 m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

fig.suptitle('First Order Finite Difference (dg/dz)', fontsize=16)
plt.savefig(os.path.join(FIG_DIR, 'First Order Finite Difference.png'))

gz_new_min = np.min(gz_new)
gz_new_max = np.max(gz_new)




#Plot 2x2 Grid
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

ax = axes[0,0]
c = ax.contourf(x_5, y_5, gz_new[:, :, 0], vmin=gz_min, vmax=gz_max, cmap="viridis_r")
ax.set_title(f'Observation level: {z_levels[0]}m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

ax = axes[0,1]
c = ax.contourf(x_5, y_5, gz_new[:, :, 1],vmin=gz_min, vmax=gz_max, cmap="viridis_r")
ax.set_title(f'Observation level: {z_levels[1]}m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

ax = axes[1,0]
c = ax.contourf(x_5, y_5, gz_new[:, :, 0], vmin=gz_new_min, vmax=gz_new_max, cmap="viridis_r")
ax.set_title(f'Observation level: {z_levels_new[0]}m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

ax = axes[1,1]
c = ax.contourf(x_5, y_5, gz_new[:, :, 1], vmin= gz_new_min, vmax=gz_new_max, cmap="viridis_r")
ax.set_title(f'Observation level: {z_levels_new[1]}m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

fig.suptitle('Gravity at 0,1,100,110m', fontsize=16)
plt.savefig(os.path.join(FIG_DIR, 'Gravity at 0,1,100,110m.png'))




#Plot Second Order Finite Difference
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

ax = axes[0]
c = ax.contourf(x_5, y_5, d2gdz2_0_laplace, levels=20, cmap='viridis_r')
ax.set_title('dg2/dz2 at dz = 0 m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

ax = axes[1]
c = ax.contourf(x_5, y_5, d2gdz2_10_laplace, levels=20, cmap='viridis_r')
ax.set_title('dg2/dz2 at dz = 10 m')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(c, ax=ax)

fig.suptitle('Second Order Finite Difference (d2g/dz2)', fontsize=16)
plt.savefig(os.path.join(FIG_DIR, 'Second Order Finite Difference.png'))