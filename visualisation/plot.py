#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def plot(fname):
    pos = np.genfromtxt(fname, delimiter=",")

    fig, ax = plt.subplots()
    xmin, xmax = 0.0, 0.0
    ymin, ymax = 0.0, 0.0

    xs = pos[:,0]
    ys = pos[:,1]

    ax.plot(xs, ys)

    xmax = max(abs(np.max(xs)), abs(np.min(xs)))
    ymax = max(abs(np.max(ys)), abs(np.min(ys)))
    xmin = -xmax
    ymin = -ymax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()


plot("test.csv")
