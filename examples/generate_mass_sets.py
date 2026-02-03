from goph547lab01.gravity import gravity_effect_point
from goph547lab01.gravity import gravity_potential_point
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat

m = 1e7

def generate_mass_set():

    masses = np.zeros(5)
    xm = np.zeros((5,3))

    pos = np.array([0,0,-10])
    mass_position = np.zeros(3)

    #Compute first 4 masses and positions
    masses[:4] = np.random.normal(m/5, m/100, 4)
    xm[:4,0] = np.random.normal(0, 20, 4)
    xm[:4,1] = np.random.normal(0, 20, 4)
    xm[:4,2] = np.random.normal(-10,2,4)

    masses[4] = m - np.sum(masses[:3])
    mass_position[0] = np.sum(masses[:4] * xm[:4,0])
    mass_position[1] = np.sum(masses[:4] * xm[:4,1])
    mass_position[2] = np.sum(masses[:4] * xm[:4,2])

    xm[4] = (m * pos - mass_position) / masses[4]

    return masses, xm

mass_set_1 = generate_mass_set()[0]
mass_set_2 = generate_mass_set()[0]
mass_set_3 = generate_mass_set()[0]
xm = generate_mass_set()[1]

savemat('mass_set_1.mat', {'mass_set_1': mass_set_1})
savemat('mass_set_2.mat', {'mass_set_2': mass_set_2})
savemat('mass_set_3.mat', {'mass_set_3': mass_set_3})