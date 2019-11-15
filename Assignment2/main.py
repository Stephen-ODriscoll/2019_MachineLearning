import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, linear_model, svm


def main(data_size, log_to_file=False):
    if log_to_file:
        import sys
        sys.stdout = open('log2.txt', 'w')
    labels, data = read_data("product_images.csv", data_size)

    title("Perceptron")
    clf1 = linear_model.Perceptron()
    perceptron_accuracy = evaluate_k_fold(labels, data, 10, clf1)

    title("Linear")
    clf2 = svm.SVC(kernel="linear")
    linear_accuracy = evaluate_k_fold(labels, data, 10, clf2)

    gammas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    title("RBF")
    best_gamma, best_accuracy = 0, 0
    for gamma in gammas:
        print("Gamma: " + str(gamma))
        clf3 = svm.SVC(kernel="rbf", gamma=gamma)
        accuracy = evaluate_k_fold(labels, data, 10, clf3)
        print("Gamma: " + str(gamma) + " achieved average Accuracy: " + str(accuracy))

        if accuracy > best_accuracy:
            best_gamma, best_accuracy = gamma, accuracy

    print()
    print("Average accuracy for Perceptron: " + str(perceptron_accuracy))
    print("Average accuracy for SVC(Linear Kernel): " + str(linear_accuracy))
    print("Best average accuracy for SVC(RBF Kernel): " + str(best_accuracy) + " with Gamma:" + str(best_gamma))


# Prints a title. e.g.
# ~~~~~~~~~
# ~ Title ~
# ~~~~~~~~~
def title(title_name):
    border = "~~~~"
    for i in range(len(title_name)):
        border += "~"
    print("\n" + border)
    print("~ " + title_name + " ~")
    print(border)


# Reads in data from the file specified, trims the data down to a user specified size and prints
# how many of each category are used then displays one image from each category of classification.
# Data must contain a label and 784 columns for pixel values (28x28). This function assumes an
# even spread (i.e. the data isn't organized making the first n elements all one category)
def read_data(file, data_size):
    width = 28
    height = 28
    data = pd.read_csv(file)
    labels, images = data["label"].iloc[:data_size], data.drop(columns="label").iloc[:data_size]

    print("Number of sneaker images: " + str(len(labels[labels == 0])))
    print("Number of ankle boot images: " + str(len(labels[labels == 1])))

    for label in range(0, 2):
        index = next(i for i in range(len(labels)) if labels.iloc[i] == label)
        plt.imshow(images.iloc[index].to_numpy().reshape(width, height), cmap='gray')
        plt.show()

    return labels, images


# Takes in labels corresponding to data (parallel arrays), number of splits and
# the classifier to use. Splits data into training and validation sets for K-Fold.
# Displays info about training time, classifying time, accuracies and confusion matrices.
# Returns average accuracy.
def evaluate_k_fold(labels, data, splits, classifier):
    train_times, predict_times, accuracies = [], [], []
    k_fold = model_selection.KFold(n_splits=splits, shuffle=False)
    for i, (train_indexes, test_indexes) in enumerate(k_fold.split(data)):
        print("\tK-fold iteration " + str(i+1) + " of " + str(splits))
        train_data, train_labels = data.iloc[train_indexes], labels.iloc[train_indexes]
        test_data, test_labels = data.iloc[test_indexes], labels.iloc[test_indexes]

        start = time()
        classifier.fit(train_data, train_labels)
        train_times.append(time()-start)

        start = time()
        prediction = classifier.predict(test_data)
        predict_times.append(time() - start)

        accuracy = round(metrics.accuracy_score(test_labels, prediction) * 100, 2)
        accuracies.append(accuracy)

        print("\t\tAccuracy: " + str(accuracy) + "%")
        print("\t\tConfusion Matrix:")

        matrix = metrics.confusion_matrix(test_labels, prediction)
        for line in matrix:
            print("\t\t\t", end="")
            print(line)

    print("\tMinimum train time: " + str(np.round(np.min(train_times), 4)) + " seconds")
    print("\tMaximum train time: " + str(np.round(np.max(train_times), 4)) + " seconds")
    print("\tAverage train time: " + str(np.round(np.average(train_times), 4)) + " seconds")
    print("\tMinimum prediction time: " + str(np.round(np.min(predict_times), 4)) + " seconds")
    print("\tMaximum prediction time: " + str(np.round(np.max(predict_times), 4)) + " seconds")
    print("\tAverage prediction time: " + str(np.round(np.average(predict_times), 4)) + " seconds")

    return np.average(accuracies)


# First parameter specifies the size of data to use.
# Pass in the second parameter True to print to log.txt. Default parameter is False
main(1000)

# Data size of 7000:
#   Accuracy:
#       SVC with RBF Kernel seems to have the best accuracy of about 97% with a gamma of 1e-06.
#       Perceptron and SVC linear kernel achieving about 94% accuracy. For accuracy alone I would
#       choose the SVC(RBF Kernel) classifier.
#
#   Times:
#       With the above parameters RBF got an average training time of 15 seconds and an average
#       classifying time of 1.6 seconds. Average training time can be brought down to 7.8 seconds with a
#       gamma of 1e-7 which achieved an average accuracy of 96%.
#       The Perceptron classifier got the best times, with an average training time of 0.3 seconds
#       and an average classification time of 0.005 seconds making it my preferred choice for classifying
#       larger more complex datasets.

# The choice between the two is a trade off of speed vs accuracy. RBF produces good accuracy but gets
# exponentially slower as a larger data size is used. With a data size of 1000, RBF took 5 times longer
# to train than the Perceptron classifier, thought still less than a second. With a data size of 7000,
# RBF took more than 20 times longer; 0.3 seconds vs 7.8 seconds. The Preceptron classifier gets a higher
# accuracy on the whole dataset (14,000), an accuracy of 95%. Given all this, due to the slightly higher
# accuracy achieved by RBF I would choose the RBF classifier over the Perceptron classifier and train
# it on a smaller dataset (size 7000). The time taken to train is an average of 15 seconds which isn't
# that much time overall, only a few minutes.
