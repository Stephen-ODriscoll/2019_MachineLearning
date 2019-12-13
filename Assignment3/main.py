import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn import model_selection


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


# Given the degree, a feature and our prediction model calculate the
# estimated target vector using a multi-variate polynomial of the specified degree
def calculate_model_function(deg, feature, p):
    result = 0
    t = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i + j + k == n:
                        result += p[t]*(feature[0]**i)*(feature[1]**j)*(feature[2]**k)
                        t += 1
    return result


# Determines the correct size for the parameter vector from the degree of the multi-variate polynomial
def num_coefficients_3(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t += 1
    return t


# Calculate target vector and the Jacobian at the linearization point
def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    j = np.zeros((1, len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        j[:, i] = di
    return f0, j


# The function takes the training target vector, the estimated target vector and Jacobian as input and
# calculates the optimal parameter update vector as output
def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


# Perform regression to generate a prediction model for how each feature effects the diamonds price
def regression(deg, features, targets):
    max_iterations = 10
    p0 = np.zeros(num_coefficients_3(deg))
    f0 = np.zeros(len(features))
    J = np.zeros((len(features), len(p0)))
    for it in range(max_iterations):
        i = 0
        targ = np.zeros(len(f0))
        for feature, target in zip(features, targets):
            f0[i], J[i, :] = linearize(deg, feature, p0)
            targ[i] = target[0]
            i += 1

        dp = calculate_update(targ, f0, J)
        p0 += dp

    return p0


# K-fold cross-validation procedure a dataset that returns the best degree
# and corresponding prediction model. Called on each dataset
def model_selection_k_fold(features, targets):
    best_overall = tuple()
    best_overall_mean = -1
    k_fold = model_selection.KFold(n_splits=5, shuffle=True)
    for iterator, (train_indexes, evaluation_indexes) in enumerate(k_fold.split(features, targets)):
        best_degree = 0
        best_mean = -1
        for degree in range(1, 4):      # 1, 2 & 3
            train_features = features.iloc[train_indexes].tolist()
            train_targets = targets.iloc[train_indexes].tolist()
            evaluation_features = features.iloc[evaluation_indexes].tolist()
            evaluation_targets = targets.iloc[evaluation_indexes].tolist()

            # Generate prediction model
            p0 = regression(degree, train_features, train_targets)
            differences = list()
            for feature, target in zip(evaluation_features, evaluation_targets):
                predicted = calculate_model_function(degree, feature, p0)           # Predict price
                differences.append(abs(predicted - target))                         # add positive difference

            mean = np.mean(differences)
            if mean < best_mean or best_mean == -1:
                best_degree = degree                # Best in this split
                best_mean = mean
                if mean < best_overall_mean or best_overall_mean == -1:
                    best_overall = (best_degree, p0)    # Best overall
                    best_overall_mean = best_mean

        print("\tSplit #" + str(iterator + 1) + ", Best Degree: " + str(best_degree) +
              ", Mean Difference: " + str(best_mean))

    return best_overall


# Plots the estimated prices against the true sale prices
def visualization(features, targets, best_overall):
    print("Visualizing predicted vs actual price with degree: " + str(best_overall[0]))
    predicted = list()
    for feature in features:
        predicted.append(calculate_model_function(best_overall[0], feature, best_overall[1]))

    plt.close("all")
    plt.title("Predicted prices vs actual prices")
    plt.plot(predicted, targets, 'b.')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def main():
    print("Reading Data")
    data = read_data("diamonds.csv")

    print("\nPerforming Regression with Degree 2")
    for iterator, subset in enumerate(data.values()):
        print("Subset #" + str(iterator + 1))
        features, targets = np.array(subset[0]), np.array(subset[1])
        p0 = regression(2, features, targets)
        print(p0)

    print("\nK-Fold and Visualization")
    for iterator, subset in enumerate(data.values()):
        print("Subset #" + str(iterator + 1))
        features, targets = pd.Series(subset[0]), pd.Series(subset[1])
        best_overall = model_selection_k_fold(features, targets)

        features, targets = np.array(subset[0]), np.array(subset[1])
        visualization(features, targets, best_overall)


main()


# I've included a file showing the output generated by the k_fold function and the graphs visualized.
# This function takes about 5 mins to run and produces different results each time due to shuffle being True.
# I've ran it a few times and the best mean differences are always within the 150 to 400 range.
# To speed up execution time and just show that the function works you can change the degree range
# from 0, 4 to 0, 3 and you should see similar results although the graph will be more scattered
# representing a worse prediction score.

# Based on the output shown in SampleOutput.txt the best overall polynomial degree to use is degree 3.
# This is the case with every subset except the second last one which shows degree 2 as better.
# The best mean differences achieved were all between 150 and 400 which is impressive
# considering some prices are in the range of 20,000 and could lend themselves to larger
# margins of error.

# The graph created by my program shows the correlation between predicted and actual price.
# The better the function the closer the graph should look to a straight line where predicted
# price increases uniformly with actual price. Sample graphs are shown in the plots folder.
