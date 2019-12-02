import numpy as np
import pandas as pd
from collections import defaultdict

def read_data(file):
    data = pd.read_csv(file)
    groups = data[["cut", "color", "clarity"]]
    features = data[["carat", "depth", "table"]]
    targets = data[["price"]]

    extracted = defaultdict(lambda: list())
    for group, feature, target in zip(groups.values, features.values, targets.values):
        extracted[tuple(group)].append((feature, target))

    meets_criteria = dict()
    for key, value in extracted.items():
        if len(value) > 800:
            meets_criteria[key] = value

    return extracted


def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t


def main():
    data = read_data("diamonds.csv")
    # print(data)


def calculate_update(y, f0, j):
    l = 1e-2
    n = np.matmul(j.T, j) + l*np.eye(j.shape[1])
    r = y-f0
    n = np.matmul(j.T, r)
    dp = np.linalg.solve


main()
