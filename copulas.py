import matplotlib.pyplot as plt


def plot_random_variable(random_variable_1, random_variable_2):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    for ax_1, ax_2, var, name in zip([ax1, ax3], [ax2, ax4], [random_variable_1, random_variable_2], ['u', 't']):
        ax_1.plot(var, 'ob')
        ax_1.set_title(f'Random variable {name}', size=15)
        count, bins, ignored = ax_2.hist(var, 20, density=False)
        ax_2.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        ax_2.set_title(f'Histogram of the {name} samples and PDF', size=15)

    plt.tight_layout()
    plt.show()


def plot_plackett(thetas: list):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    axes = [ax1, ax2, ax3]

    for theta, ax in zip(thetas, axes):
        a = t * (1 - t)
        b = theta + a * (theta - 1) ** 2
        c = 2 * a * (u * (theta ** 2) + 1 - u) + theta * (1 - 2 * a)
        d = (theta ** 0.5) * (theta + 4 * a * u * (1 - u) * (1 - theta) ** 2) ** 0.5
        v = (c - (1 - 2 * t) * d) / (2 * b)

        points: list = []
        for i in range(len(u)):
            points.append((u[i], v[i]))

        x, y = zip(*points)
        ax.scatter(x, y)
        ax.set_title(f'theta = {theta}', size=20)

    plt.tight_layout()
    plt.show()


def plot_amh(thetas: list):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    axes = [ax1, ax2, ax3]

    for theta, ax in zip(thetas, axes):
        a = 1 - u
        b = - theta * (2 * a * t + 1) + 2 * (theta ** 2) * (a ** 2) * t + 1
        c = (theta ** 2) * (4 * (a ** 2) * t - 4 * a * t + 1) - theta * (4 * a * t - 4 * t + 2) + 1
        v = (2 * t * ((a * theta - 1) ** 2)) / (b + (c ** 0.5))

        points: list = []
        for i in range(len(u)):
            points.append((u[i], v[i]))

        x, y = zip(*points)
        ax.scatter(x, y)
        ax.set_title(f'theta = {theta}', size=20)

    plt.tight_layout()
    plt.show()
