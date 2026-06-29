import numpy as np
import matplotlib.pyplot as plt

from nbody import *

def test_calc_stable_orbit():

    (pos, vel) = calc_stable_orbit(np.ones(1), np.zeros(1))

    np.testing.assert_almost_equal( pos, np.array([0., 1.]).reshape(1,2) )
    np.testing.assert_almost_equal( vel, np.array([-1., 0.]).reshape(1,2) )

    (pos2, vel2) = calc_stable_orbit(np.ones(1), np.ones(1) * 2.0 * np.pi)

    np.testing.assert_almost_equal( pos, pos2 )
    np.testing.assert_almost_equal( vel, vel2 )

    (pos3, vel3) = calc_stable_orbit(np.ones(1), np.ones(1) * np.pi)

    np.testing.assert_almost_equal( pos, pos3 * -1.0 )
    np.testing.assert_almost_equal( vel, vel3 * -1.0 )


def test_calc_acc():

    mass = np.array([2.0,0.5])
    pos = np.zeros((2, 2))
    acc = np.zeros_like(pos)

    pos[0,:] = np.array([0, 0])
    pos[1,:] = np.array([1, 0])
    
    calc_acc(acc, pos, mass)
    epsilon = 1.1*np.power(len(pos), -0.48)
    np.testing.assert_almost_equal(acc[0], np.array([1, 0]) * mass[1] * (1 + epsilon**2)**-1.5)
    np.testing.assert_almost_equal(acc[1], -np.array([1, 0]) * mass[0] * (1 + epsilon**2)**-1.5)

    pos[0,:] = np.array([0, 0])
    pos[1,:] = np.array([0, 1])
    
    calc_acc(acc, pos, mass)
    epsilon = 1.1*np.power(len(pos), -0.48)
    np.testing.assert_almost_equal(acc[0], np.array([0, 1]) * mass[1] * (1 + epsilon**2)**-1.5)
    np.testing.assert_almost_equal(acc[1], -np.array([0, 1]) * mass[0] * (1 + epsilon**2)**-1.5)


def test_advance_pos():

    pos = np.zeros((1, 2))
    pos_prev = np.zeros_like(pos)
    pos_temp = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    dt = 0.5;

    pos[0,:] = np.array([1, 2])
    pos_prev[0,:] = np.array([0.5, 3])
    acc[0,:] = np.array([0.5, -1])

    advance_pos(acc, pos, pos_prev, pos_temp, dt)
    
    np.testing.assert_almost_equal(pos[0], np.array([\
        2 - 0.5 + 0.5 * 0.5**2,\
        4 - 3 + -1*0.5**2\
    ]))
    

test_calc_acc()
test_advance_pos()
