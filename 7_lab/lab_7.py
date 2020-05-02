import numpy as np
from scipy.stats import norm, chi2

print("---------------------------------------NORMAL----------------------------------------------------------")


# NORMAL
def MLE(x):
    mu = np.mean(x)
    sigma = np.std(x)
    return mu, sigma


def hiN(x, k, step, a0):
    size = x.size
    n = []
    p = []
    a = []
    np = []
    nnp = []
    hi = []

    ni = x[x <= a0].size
    pi = norm.cdf(a0)
    npi = size * pi
    nnpi = ni - npi
    hii = nnpi ** 2 / npi

    n.append(ni)
    p.append(pi)
    np.append(npi)
    nnp.append(nnpi)
    hi.append(hii)

    for i in range(k - 2):
        a1 = a0 + step
        ni = x[(x > a0) & (x <= a1)].size
        pi = norm.cdf(a1) - norm.cdf(a0)
        npi = size * pi
        nnpi = ni - npi
        hii = nnpi ** 2 / npi

        n.append(ni)
        p.append(pi)
        np.append(npi)
        nnp.append(nnpi)
        hi.append(hii)
        a.append(a0)

        a0 += step

    ni = x[x > a0].size
    pi = 1 - norm.cdf(a0)
    npi = size * pi
    nnpi = ni - npi
    hii = nnpi ** 2 / npi

    n.append(ni)
    p.append(pi)
    np.append(npi)
    nnp.append(nnpi)
    hi.append(hii)
    a.append(a0)

    return a, n, p, np, nnp, hi


def main():
    size = 100
    k = 7
    alpha = 0.05
    step = 0.45
    a0 = -1

    x = norm.rvs(loc=0, scale=1, size=size)
    x.sort()
    h = chi2.ppf(1 - alpha, k - 1)

    mu, sigma = MLE(x)
    a, n, p, np, nnp, hii = hiN(x, k, step, a0)

    print('mu = ', mu, 'sigma = ', sigma)
    print('a = ', a)
    print('n = ', n)
    print('p = ', p)
    print('np = ', np)
    print('nnp = ', nnp)
    print('hii = ', hii)
    print('sum(n) =  ', sum(n))
    print('sum(p) = ', sum(p))
    print('sum(np) = ', sum(np))
    print('sum(nnp) = ', sum(nnp))
    print('sum(hii) = ', sum(hii))
    print('hi = ', h)


if __name__ == "__main__":
    main()
