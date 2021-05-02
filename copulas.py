import matplotlib.pyplot as plt
import numpy as np


def plot_random_variable(random_variable_1, random_variable_2, random_variable_3=None):
    if random_variable_3 is None:
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        zip_statement = zip([ax1, ax3], [ax2, ax4], [
                            random_variable_1, random_variable_2], ['u', 't'])
    elif random_variable_3 is not None:
        _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))
        zip_statement = zip([ax1, ax3, ax5], [ax2, ax4, ax6], [random_variable_1,
                                                               random_variable_2, random_variable_3], ['r', 's', 't'])

    for ax_1, ax_2, var, name in zip([ax1, ax3], [ax2, ax4], [random_variable_1, random_variable_2], ['u', 't']):
        ax_1.plot(var, 'ob')
        ax_1.set_title(f'Random variable {name}', size=15)
        count, bins, ignored = ax_2.hist(var, 20, density=False)
        ax_2.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        ax_2.set_title(f'Histogram of the {name} samples and PDF', size=15)

    plt.tight_layout()
    plt.show()


def plot_contour(copula_function):
    u = np.linspace(0, 1, 10)
    v = np.linspace(0, 1, 10)
    U, V = np.meshgrid(u, v)
    Z = copula_function(U, V)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(U, V, Z, 50, cmap='viridis')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('z')
    ax.set_title('Marshall-Olkin copula')

    plt.tight_layout()
    plt.show()


def plot_surface(copula_function):
    u = np.linspace(0, 1, 10)
    v = np.linspace(0, 1, 10)
    U, V = np.meshgrid(u, v)
    Z = copula_function(U, V)

    ax = plt.axes(projection='3d')
    ax.plot_surface(U, V, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='black')
    ax.set_title('Marshall-Olkin copula')

    plt.tight_layout()
    plt.show()


def plot_mo(lambdas, r, s, t):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    axes = [ax1, ax2, ax3]

    for lambd, ax in zip(lambdas, axes):
        x = np.minimum(-(np.log(r) / lambd[0]), -(np.log(t) / lambd[2]))
        y = np.minimum(-(np.log(s) / lambd[1]), -(np.log(t) / lambd[2]))

        u = np.exp(-(lambd[0] + lambd[2]) * x)
        v = np.exp(-(lambd[1] + lambd[2]) * y)
        ax.scatter(u, v)
        ax.set_title(f'lambda1 = {lambd[0]}, lambda2 = {lambd[1]}, lambda12 = {lambd[2]}')

    plt.tight_layout()
    plt.show()


def plot_amh(thetas: list, u, t):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    axes = [ax1, ax2, ax3]

    for theta, ax in zip(thetas, axes):
        a = 1 - u
        b = - theta * (2 * a * t + 1) + 2 * (theta ** 2) * (a ** 2) * t + 1
        c = (theta ** 2) * (4 * (a ** 2) * t - 4 * a * t + 1) - theta * (4 * a * t - 4 * t + 2) + 1
        v = (2 * t * ((a * theta - 1) ** 2)) / (b + (c ** 0.5))

        ax.scatter(u, v)
        ax.set_title(f'theta = {theta}', size=20)

    plt.tight_layout()
    plt.show()
