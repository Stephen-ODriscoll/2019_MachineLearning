import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, linear_model, svm


def main(log_to_file=False):
    if log_to_file:
        import sys
        sys.stdout = open('log2.txt', 'w')

    data_size = 7000
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


def title(title_name):
    border = "~~~~"
    for i in range(len(title_name)):
        border += "~"
    print("\n" + border)
    print("~ " + title_name + " ~")
    print(border)


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


# Pass in the value true to print to log.txt
main()
