import sys
import numpy as np

r_caff = 1 / 4
bou = np.sqrt(3)

def norm_distr(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)


def laplace_distr(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))


def cauchy_distr(x):
    return 1 / (np.pi * (1 + x * x))


def poisson_distr(x):
    return (np.power(x, 10) / np.math.factorial(10)) * np.exp(-x)


def uniform_distr(x):
    return 1 / (2 * bou) * (x <= bou)


def laplace_gen(x):
    return np.random.laplace(0, 1 / np.sqrt(2), x)


def poisson_gen(x):
    return np.random.poisson(10, x)


def uniform_gen(x):
    return np.random.uniform(-bou, bou, x)


def Ztr(x):
    length = x.size
    R = (int)(length * r_caff)
    summ = 0
    for i in range(R, length - R):
        summ += x[i]
    return summ / (length - 2 * R)


def Zr(x):
    return (np.amax(x) + np.amin(x)) / 2


def Zq(x):
    return (np.quantile(x, r_caff) + np.quantile(x, 3 * r_caff)) / 2


distrs = {
    'normal': norm_distr,
    'cauchy': cauchy_distr,
    'laplace': laplace_distr,
    'poisson': poisson_distr,
    'uniform': uniform_distr,
}

generate_dict = {
    'normal': np.random.standard_normal,
    'cauchy': np.random.standard_cauchy,
    'laplace': laplace_gen,
    'poisson': poisson_gen,
    'uniform': uniform_gen,
}



pos_characteristic_dict = {
    'average': np.mean,
    'med': np.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr r = n/4': Ztr
}

pos_names = [
    'average',
    'med',
    'Zr',
    'Zq',
    'Ztr r = n/4'
]


def E(z):
    return np.mean(z)


def D(z):
    return np.var(z)


selections = [10, 50, 1000]
sys.stdout = open('write.txt', 'w')


def research(dist_type):
    print('-------------------------------------')
    print(dist_type)
    for selection in selections:
        print_table = {
            'E': [],
            'D': []
        }
        for name in pos_names:
            z = []
            for i in range(0, 1000):
                dates = np.sort(generate_dict[dist_type](selection))
                z.append(pos_characteristic_dict[name](dates))
            print_table['E'].append(E(z))
            print_table['D'].append(D(z))

        print_list(selection, pos_names)
        printED('E =', print_table['E'])
        printED('D =', print_table['D'])

        print()


def print_list(num, lis):
    print()
    print("%-9s" % ('n = %i' % num), end="")
    for i in lis:
        print("%-17s" % i, end="")


def printED(ED, printTableED):
    print()
    print("%-9s" % (ED), end="")
    for e in printTableED:
        print("%-17f" % e, end="")


def main():
    research('normal')
    research('cauchy')
    research('laplace')
    research('poisson')
    research('uniform')


if __name__ == "__main__":
    main()

