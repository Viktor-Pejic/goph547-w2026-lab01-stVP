from scipy.io import loadmat
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
