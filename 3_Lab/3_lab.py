import numpy
import matplotlib.pyplot as plot
import sys
import seaborn as sns

POISSON_P = 10
UNIFORM_L = -numpy.sqrt(3)
UNIFORM_R = numpy.sqrt(3)
LAPLAS_C = numpy.sqrt(2)
selection = [20, 100]


def standart_normal(x):
    return (1 / numpy.sqrt(2 * numpy.pi)) * numpy.exp(- x * x / 2)


def standart_cauchy(x):
    return 1 / (numpy.pi * (1 + x * x))


def laplace(x):
    return 1 / LAPLAS_C * numpy.exp(-LAPLAS_C * numpy.abs(x))


def poisson(x):
    return (numpy.power(x, POISSON_P) / numpy.math.factorial(POISSON_P)) * numpy.exp(-x)


def uniform(x):
    sign2 = x <= UNIFORM_R
    sign1 = x >= UNIFORM_L
    return 1 / (UNIFORM_R - UNIFORM_L) * sign1 * sign2


func_dict = {
    'normal': standart_normal,
    'cauchy': standart_cauchy,
    'laplace': laplace,
    'uniform': uniform,
    'poisson': poisson
}


def generate_laplace(x):
    return numpy.random.laplace(0, 1 / LAPLAS_C, x)


def generate_uniform(x):
    return numpy.random.uniform(UNIFORM_L, UNIFORM_R, x)


def generate_poisson(x):
    return numpy.random.poisson(POISSON_P, x)


generate_dict = {
    'normal': numpy.random.standard_normal,
    'cauchy': numpy.random.standard_cauchy,
    'laplace': generate_laplace,
    'uniform': generate_uniform,
    'poisson': generate_poisson
}


def Zr(x):
    return (numpy.amin(x) + numpy.amax(x)) / 2


def Zq(x):
    return (numpy.quantile(x, 1 / 4) + numpy.quantile(x, 3 / 4)) / 2


def Ztr(x):
    length = x.size
    r = (int)(length / 4)
    count = 0
    for i in range(r, length - r):
        count += x[i]
    return count / (length - 2 * r)


def IQR(x):
    return numpy.abs(numpy.quantile(x, 1 / 4) - numpy.quantile(x, 3 / 4))


def ejection(x):
    length = x.size
    count = 0
    left = numpy.quantile(x, 1 / 4) - 1.5 * IQR(x)
    right = numpy.quantile(x, 3 / 4) + 1.5 * IQR(x)
    for i in range(0, length):
        if (x[i] < left or x[i] > right):
            count += 1
    return count / length


pos_characteristic_dict = {
    'average': numpy.mean,
    'med': numpy.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr r = n/4': Ztr
}

pos_char_name = [
    'average',
    'med',
    'Zr',
    'Zq',
    'Ztr r = n/4'
]


def E(z):
    return numpy.mean(z)


def D(z):
    return numpy.var(z)


def research(dist_type):
    # print('-------------------------------------')
    print()
    print(dist_type)

    data = []

    for num in selection:
        fict = []
        arr = numpy.sort(generate_dict[dist_type](num))
        data.append(arr)

        for i in range(0, 1000):
            arr = numpy.sort(generate_dict[dist_type](num))
            fict.append(ejection(arr))

        print("%-10s;" % ('n = %i' % num), end="")
        print("%-12f;" % E(fict), end="")
        print()

    plot.figure(dist_type)
    plot.title(dist_type)
    sns.set(style="whitegrid")
    axis = sns.boxplot(data=data, orient='h')
    plot.yticks(numpy.arange(2), ('20', '100'))
    plot.show()


def main():
    stream = open('out_file.csv', 'w')
    std = sys.stdout
    sys.stdout = stream

    research('normal')
    research('cauchy')
    research('laplace')
    research('poisson')
    research('uniform')

    stream.close()
    sys.stdout = std
    print("Done")


if __name__ == "__main__":
    main()
