import numpy as np
from numpy.random import default_rng
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from time import perf_counter

from numba import njit, prange


PARALLEL=False
N_YEARS = 0.1
N_PARTICLES = 100
FP_TYPE = np.float64


# L_MOON = L_SATURN # m
# T_MOON = 1e8 # s
# M_MOON = 4e19 # kg
# G_MOON = G_MKS * M_MOON * T_MOON**2 / L_MOON**3

# EPSILON_MOON = 250e3/L_MOON # radius of Enceladus

# print(G_MOON)


def calc_grav_const(L, T, M):
    G_MKS = 6.67408e-11 # m**3 kg**-1 s**-2
    return G_MKS * M * T**2 / L**3


class SaturnSystem():
    def __init__(self, L, T, M, particle_pos, particle_vel, dt):
        self.G = calc_grav_const(L, T, M)
        self.L = L
        self.T = T
        self.M = M
        self.EPSILON = 58e6/self.L
        self.dt = dt

        self.particle_pos = particle_pos.copy()
        self.acc = np.zeros_like(self.particle_pos)
        self.body_pos = np.array([(0.,0.)])
        self.body_mass = np.array([5.683e26/self.M])

        self.calc_acc()
        self.particle_pos_prev = self.particle_pos - particle_vel*self.dt - 0.5*self.acc*self.dt**2
        self.particle_pos_temp = np.zeros_like(self.particle_pos)

    @njit
    def calc_acc(self):
        for i in prange(len(particle_pos)):
            acc[i,:] = 0.0
            for j in range(len(body_pos)):
                r = body_pos[j,:] - particle_pos[i,:]
                acc[i,:] += G * r * body_mass[j] / (r[0]**2 + r[1]**2 + epsilon**2)**(1.5)

    def advance_pos(self):
        advance_pos(self.acc, self.particle_pos, self.particle_pos_prev, self.particle_pos_temp, self.dt)

    def plot(self):
        plt.scatter(self.particle_pos[:,0], self.particle_pos[:,1])
        plt.quiver(
            self.particle_pos[:,0],
            self.particle_pos[:,1],
            (self.particle_pos[:,0]-self.particle_pos_prev[:,0])/(2*self.dt),
            (self.particle_pos[:,1]-self.particle_pos_prev[:,1])/(2*self.dt))
        plt.show()



def random(num, a=0., b=1.):
    rng = default_rng()
    return rng.random(num, dtype=FP_TYPE)*(b-a) + a


def calc_stable_orbit(r, theta, GM=1.):
    v_mag = np.sqrt(GM/r)

    pos = np.zeros((r.shape[0], 2))
    vel = np.zeros((r.shape[0], 2))

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    return pos, vel


def generate_ring(num, radius_min, radius_max, GM=1.):
    # Generate similar system to the solar system
    r = random(num, radius_min, radius_max)

    theta = random(num, 0., 2.*np.pi)
    pos, vel = calc_stable_orbit(r, theta, GM=GM)

    return pos, vel


@njit(parallel=True)
def calc_acc(acc, pos, mass, epsilon=0.01):
    # Calc acceleration on particles from all other particles
    for i in prange(len(particle_pos)):
        acc[i,:] = 0.0
        for j in range(len(particle_pos)):
            if i == j:
                continue
            r = pos[j] - pos[i]
            acc[i,:] += r * mass[j] / (r[0]**2 + r[1]**2 + epsilon**2)**(1.5)


# @njit
def advance_pos(acc, pos, pos_prev, pos_temp, dt):
    pos_temp[:] = pos[:]
    pos[:] = 2.0 * pos - pos_prev + acc * dt**2
    pos_prev[:] = pos_temp[:]


def main():
    dt = 0.0001
    total_time = 10000*dt
    # total_time = N_YEARS*2*np.pi

    # Load initial conditions
    # pos, vel, mass = create_solar_system()
    ring_pos, ring_vel = generate_ring(N_PARTICLES, 1, 2, GM=1.*calc_grav_const(1e8, 1e5, 5.683e26))
    saturnSystem = SaturnSystem(1e8, 1e5, 5.683e26, ring_pos, ring_vel, dt)

    start = perf_counter()

    t = 0
    while t < total_time:
        saturnSystem.calc_acc()
        saturnSystem.advance_pos()
        t += dt

    end = perf_counter()

    print(f"Time to complete: {end-start}")

    saturnSystem.plot()

    # positions_for_plotting = np.array(pos_tracker)

    # ax = plt.axes(projection='3d')
    # ax = plt.axes()
    # xmin, xmax = 0.0, 0.0
    # ymin, ymax = 0.0, 0.0
    # for i in range(len(pos)):
        # xdata = positions_for_plotting[:,i,0]
        # ydata = positions_for_plotting[:,i,1]
        # xmin = min(np.min(xdata), xmin)
        # xmax = max(np.max(xdata), xmax)
        # ymin = min(np.min(ydata), ymin)
        # ymax = max(np.max(ydata), ymax)
        # ax.plot(xdata, ydata)

    # xmax = max(abs(xmax), abs(xmin))
    # ymax = max(abs(ymax), abs(ymin))
    # xmin = -xmax
    # ymin = -ymax
    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
    # for N_PARTICLES in [8, 16, 32, 64, 128, 256, 512]:
        # print(N_PARTICLES)
        # main()
