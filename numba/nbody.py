#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from time import perf_counter
from numba import cuda
import nvtx
import math


N_YEARS = 0.1
FP_TYPE = np.float64

def random(num, a=0., b=1.):
    """
    Generate a random number between a and b
    """
    rng = default_rng()
    return rng.random(num, dtype=FP_TYPE)*(b-a) + a

def calc_stable_orbit(r, theta):
    """
    Given a position (in polar coords) calculate the velocity required to keep the body in a stable, circular orbit around the origin. Also returns Cartesian positions.

    Args:
        r: list of radii (one per body)
        theta: list of angles (one per body)

    Returns:
        List of Cartesian positions and velocities
    """
    v_mag = 1.0 / np.sqrt(r)

    pos = np.zeros((len(r),2),dtype=FP_TYPE)
    vel = np.zeros((len(r),2),dtype=FP_TYPE)

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    return pos,vel

def generate_random_star_system(num,min_radius=0.4,max_radius=20,min_mass=1./6000000,max_mass=1./1000):
    """
    Generate positions, velocities and masses for a star system similar to our own. Assumes star is massive (with mass=1) with zero velocity at coords (0,0).

    Args:
        num: Number of bodies to simulate (including central star)
        min_radius: sets the minimum radius possible
        max_radius: sets the maximum radius possible
        min_mass: sets the minimum mass possible
        max_mass: sets the maximum mass possible

    Returns:
        Lists of positions, velocities and masses, one per body
    """
    r = random(num,min_radius,max_radius)
    mass = random(num,min_mass,max_mass)

    theta = random(num, 0., np.pi)
    pos,vel = calc_stable_orbit(r, theta)

    # Add central star
    pos[0] = 0
    vel[0] = 0
    mass[0] = 1.0

    return pos, vel, mass


def create_solar_system():
    """
    Generate positions, velocities and masses for our own solar system. Assumes star is massive (with mass=1) with zero velocity at coords (0,0).
    """

    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data

    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune", "Uranus"]

    mass = np.array((1,1/6023600,1/408524,1/332946.038,1/3098710,1/1047.55,1/3499,1/22962,1/19352))
    r = np.array((0.1, 0.4, 0.7, 1, 1.5, 5.2, 9.5, 19.2, 30.1))
    theta = random(len(r), 0., np.pi)
    pos, vel = calc_stable_orbit(r, theta)

    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass

@cuda.jit
def calc_acc_kernel(acc,pos,mass,epsilon):
    """
    Accumulate gravitational forces and calculate acceleration. This uses a very simple method which directly calculates the gravitational interaction between every single pair of bodies.
    Each CUDA thread computes the acceleration for one particle.
    Args:
        acc: array to be updated with new accelerations
        pos: current positions of all bodies
        mass: masses of all bodies
    """

    i = cuda.grid(1)

    if i >= pos.shape[0]:
        return

    xi = pos[i, 0]
    yi = pos[i, 1]

    ax = 0.0
    ay = 0.0

    # Loop over every particle in the system
    for j in range(pos.shape[0]):
        dx = pos[j, 0] - xi
        dy = pos[j, 1] - yi
        dist = dx * dx + dy * dy + epsilon * epsilon
        inv_r3 = 1.0 / (dist * math.sqrt(dist))
        ax += dx * mass[j] * inv_r3
        ay += dy * mass[j] * inv_r3
    acc[i, 0] = ax
    acc[i, 1] = ay

@cuda.jit
def advance_pos_kernel(pos, pos_prev, acc, dt):
    """
    Advance positions of all bodies based on previous position and current acceleration. This uses the Verlet method which is useful for simulations of simple equations of motion. It is 4th-order accurate in dt (compared to only 1st-order for the Euler method) making it suitable for longer-running simulations like this one.
    Each CUDA thread updates one particle.
    Args:
        acc: list of accelerations
        pos: current positions of all bodies
        pos_prev: previous positions of all bodies
        pos_temp: array to temporarily hold positions during calculation
        dt: simulation timestep
    """

    i = cuda.grid(1)

    if i >= pos.shape[0]:
        return

    x = pos[i, 0]
    y = pos[i, 1]

    prev_x = pos_prev[i, 0]
    prev_y = pos_prev[i, 1]

    ax = acc[i, 0]
    ay = acc[i, 1]

    pos_prev[i, 0] = x
    pos_prev[i, 1] = y

    pos[i, 0] = 2.0 * x - prev_x + ax * dt * dt
    pos[i, 1] = 2.0 * y - prev_y + ay * dt * dt

def run(is_solar_system=False, plot=False, n_particles=2000, steps=None,):
    """
    Run N-body simulation.
    """
    print(f"N-body simulation (N = {n_particles})")

    dt = 0.01

    if steps is None:
        total_time = N_YEARS * 2.0 * np.pi
    else:
        total_time = steps * dt

    if is_solar_system:
        pos, vel, mass = create_solar_system()
    else:
        pos, vel, mass = generate_random_star_system(n_particles)

    epsilon = 1.1 * n_particles ** (-0.48)

    threads = 256
    blocks = (n_particles + threads - 1) // threads

    # Allocate GPU memory
    d_pos = cuda.to_device(pos)
    d_mass = cuda.to_device(mass)
    d_acc = cuda.device_array_like(d_pos)
    # Previous positions for Verlet
    pos_prev = pos - vel * dt

    d_pos_prev = cuda.to_device(pos_prev)

    pos_tracker = []

    calc_acc_kernel[blocks, threads](d_acc, d_pos, d_mass, epsilon,)
    cuda.synchronize()

    with nvtx.annotate("NBody Simulation"):
        start = perf_counter()
        current_time = 0.0
        step = 0
        while current_time < total_time:
            if plot:
                pos_tracker.append(d_pos.copy_to_host())
            # Compute accelerations
            with nvtx.annotate("Acceleration"):
                calc_acc_kernel[blocks, threads](d_acc, d_pos, d_mass, epsilon,)
            with nvtx.annotate("Verlet Update"):
                advance_pos_kernel[blocks, threads](d_pos, d_pos_prev, d_acc, dt,)
            current_time += dt
            step += 1

        cuda.synchronize()

        end = perf_counter()

    completion_time = end - start

    print(f"Steps   : {step}")
    print(f"Time to complete: {completion_time:.4f} s")

    if plot:
        positions_for_plotting = np.array(pos_tracker)
        
        fig, ax = plt.subplots()
        xmin, xmax = 0.0, 0.0
        ymin, ymax = 0.0, 0.0
        for i in range(len(pos)):
            xdata = positions_for_plotting[:, i, 0]
            ydata = positions_for_plotting[:, i, 1]
            xmin = min(np.min(xdata), xmin)
            xmax = max(np.max(xdata), xmax)
            ymin = min(np.min(ydata), ymin)
            ymax = max(np.max(ydata), ymax)
            ax.plot(xdata, ydata)

        xmax = max(abs(xmax), abs(xmin))
        ymax = max(abs(ymax), abs(ymin))
        xmin = -xmax
        ymin = -ymax
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
        plt.show()

    return completion_time