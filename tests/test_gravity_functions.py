from goph547lab01.gravity import gravity_effect_point
from goph547lab01.gravity import gravity_potential_point
import numpy as np

def test_gravity_potential_point():
    x = [1,2,3]
    xm = [4,5,6]
    m = 1e7

    U = gravity_potential_point(x, xm, m, G=6.674e-11)
    print(f'Gravity at a potential point: {U}')

def test_gravity_effect_point():
    x = [1,2,3]
    xm = [4,5,6]
    m = 1e7

    gz = gravity_effect_point(x, xm, m, G=6.674e-11)
    print(f'Gravity at an effect point: {gz}')

def main():
    test_gravity_potential_point()
    test_gravity_effect_point()

if __name__ == '__main__':
    main()