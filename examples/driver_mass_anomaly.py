from scipy.io import loadmat
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
axes[2].set_xlim(20,-20)
axes[2].set_ylim(25,-25)
axes[2].set_title('Anomaly in xy-plane')
cbar2 = fig.colorbar(c0, ax=axes[2])



plt.tight_layout()
#plt.show()