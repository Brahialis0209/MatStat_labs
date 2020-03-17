import numpy as np
import matplotlib.pyplot as pl


# ---------Нормальное распределение------------
def normalized_distribution(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)

# --------- Распределение Лапласса ------------
def laplace_distribution(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))

# --------- Распределение Коши ------------
def cauchy_distribution(x):
    return 1 / (np.pi * (1 + x * x))

# --------- Распределение Пуассона ------------
def poisson_distribution(x):
    return (np.power(x, lm) / np.math.factorial(lm)) * np.exp(-x)

# --------- Равномерное распределение ------------
def uniform_distribution(x):
    return 1 / (2 * np.sqrt(3)) * (x <= np.sqrt(3))


def laplace_generation(x):
    return np.random.laplace(0, 1 / np.sqrt(2), x)


def poisson_generation(x):
    return np.random.poisson(lm, x)


def uniform_generation(x):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), x)


distrs_dict = {
    'Normal': normalized_distribution,
    'Cauchy': cauchy_distribution,
    'Laplace': laplace_distribution,
    'Poisson': poisson_distribution,
    'Uniform': uniform_distribution,
}

generate_dict = {
    'Normal': np.random.standard_normal,
    'Cauchy': np.random.standard_cauchy,
    'Laplace': laplace_generation,
    'Poisson': poisson_generation,
    'Uniform': uniform_generation,
}


def draw_gr(func: str, iter: int, count: int):
    pl.subplot(iter)
    pl.tight_layout()
    gen_list = generate_dict[func](count)
    pl.hist(gen_list, 17, density=True)
    ab = np.linspace(np.min(gen_list), np.max(gen_list), 95)
    pl.plot(ab, distrs_dict[func](ab), 'g')
    pl.title('N = ' + str(count))


def start_work(distrs_count: list, distr: str):
    iter = 222
    for count in distrs_count:
        draw_gr(distr, iter, count)
        iter += 1
    pl.show()


lm = 7

if __name__ == "__main__":
    distrs_name = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
    distrs_count = [10, 100, 1000]
    for distr in distrs_name:
        pl.figure("Distribution - " + distr)
        start_work(distrs_count, distr)
