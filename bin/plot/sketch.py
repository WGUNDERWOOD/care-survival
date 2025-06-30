import numpy as np
import matplotlib.pyplot as plt
import common

figsize = (2.4, 2.5)
common.init()


def plot_gaussian():
    y = 0
    a = 1
    xs = np.arange(-2, 2, 0.02)
    ks = a + np.exp(-(xs - y) * (xs - y))
    (fig, ax) = plt.subplots(figsize=figsize)
    plt.plot(xs, ks, c="k", lw=1)
    plt.xlabel("Input $x$")
    plt.ylabel("Kernel")
    plt.savefig("plot/sketch_kernel_gaussian.pdf", bbox_inches="tight")
    plt.close("all")


def plot_polynomial():
    y = 2
    a = 1
    p = 3
    xs = np.arange(-2, 2, 0.02)
    ks = (xs * y + a) ** p
    (fig, ax) = plt.subplots(figsize=figsize)
    plt.plot(xs, ks, c="k", lw=1)
    plt.xlabel("Input $x$")
    plt.ylabel("Kernel")
    plt.savefig("plot/sketch_kernel_polynomial.pdf", bbox_inches="tight")
    plt.close("all")


def plot_sobolev():
    y = 0
    a = 1
    xs = np.arange(-2, 2, 0.02)
    ks = [a + min(x, y) for x in xs]
    (fig, ax) = plt.subplots(figsize=figsize)
    plt.plot(xs, ks, c="k", lw=1)
    plt.xlabel("Input $x$")
    plt.ylabel("Kernel")
    plt.yticks([-1, 0, 1])
    plt.savefig("plot/sketch_kernel_sobolev.pdf", bbox_inches="tight")
    plt.close("all")


plot_gaussian()
plot_polynomial()
plot_sobolev()
