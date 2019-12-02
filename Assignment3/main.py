import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def read_data(file):
    data = pd.read_csv(file)
    groups = data[["cut", "color", "clarity"]]
    features = data[["carat", "depth", "table"]]
    targets = data[["price"]]

    # Creates a dictionary where key is a tuple of cut, color and clarity.
    # Value is a tuple of two lists; features and targets.
    extracted = defaultdict(lambda: (list(), list()))
    for group, feature, target in zip(groups.values, features.values, targets.values):
        extracted[tuple(group)][0].append(feature)
        extracted[tuple(group)][1].append(target)

    meets_criteria = dict()
    for key, value in extracted.items():
        if len(value[0]) > 800:
            meets_criteria[key] = value

    return meets_criteria


def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    j = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        j[:, i] = di
    return f0, j


def calculate_model_function(deg, data, p):
    result = np.zeros(data.shape[0])
    t = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i + j + k == n:
                        result += p[t]*(data[:, 0]**i)*(data[:, 1]**j)*(data[:, 2]**k)
                        t += 1
    return result


def num_coefficients_3(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t = t + 1
    return t


def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


def main():
    data = read_data("diamonds.csv")
    max_iter = 10
    for subset in data.values():
        features, target = np.array(subset[0]), np.array(subset[1])
        for deg in range(5):
            p0 = np.zeros(num_coefficients_3(deg))
            for i in range(max_iter):
                f0, j = linearize(deg, features, p0)
                dp = calculate_update(target, f0, j)
                print(str(len(p0)) + " " + str(len(dp)))
                p0 += dp

            x, y = np.meshgrid(np.arange(np.min(features[:, 0]), np.max(features[:, 0]), 0.1),
                               np.arange(np.min(features[:, 1]), np.max(features[:, 1]), 0.1))
            test_data = np.array([x.flatten(), y.flatten()]).transpose()
            test_target = calculate_model_function(deg, test_data, p0)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features[:, 0], features[:, 1], target, c='r')
            ax.plot_surface(x, y, test_target.reshape(x.shape))
            plt.show()


main()
