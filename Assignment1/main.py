import re
import numpy as np
import pandas as pd
from math import log
from sklearn import metrics, model_selection


def main(min_length=3, min_occurrence=50):
    print("Min length: " + str(min_length) + ", Min occurrence: " + str(min_occurrence))

    print("Reading data")
    train_data, train_labels, test_data, test_labels = read_data("movie_reviews.xlsx")

    print("Counting occurrences of words")
    unique_words = get_unique_words(train_data, min_length, min_occurrence)

    print("Counting frequencies of words in reviews")
    pos_freq_dict, neg_freq_dict = count_review_frequencies(train_data, train_labels, unique_words)

    print("Generating likelihood")
    pos_prob_dict, neg_prob_dict, pos_prior, neg_prior = likelihood_present(pos_freq_dict, neg_freq_dict, train_labels)

    print("Classifying evaluation data")
    predicted_labels = predict_labels(test_data, pos_prob_dict, neg_prob_dict, pos_prior, neg_prior)

    check_accuracy(test_labels, predicted_labels)

    print("\nEvaluating k-fold")
    evaluate_k_fold(train_data, train_labels, min_length, min_occurrence, 10)


def read_data(file):
    data = pd.read_excel(file)
    training = data[data["Split"] == "train"]
    evaluation = data[data["Split"] == "test"]

    print("Number of positive training reviews: " + str(len(training[training["Sentiment"] == "positive"])))
    print("Number of negative training reviews: " + str(len(training[training["Sentiment"] == "negative"])))
    print("Number of positive evaluation reviews: " + str(len(evaluation[evaluation["Sentiment"] == "positive"])))
    print("Number of negative evaluation reviews: " + str(len(evaluation[evaluation["Sentiment"] == "negative"])))

    return training["Review"].to_list(), training["Sentiment"].to_list(),\
        evaluation["Review"].to_list(), evaluation["Sentiment"].to_list()


def divide_set(data, size):
    return (data[i::size] for i in range(size))


def clean_and_split(data):
    return re.sub(r'[^\w]', ' ', data).lower().split()


def get_unique_words(data, min_length, min_occurrence):
    words = []
    for review in data:
        words += clean_and_split(review)

    occurrences = dict.fromkeys(words, 0)
    for word in words:
        if len(word) > min_length:
            count = occurrences.get(word)
            occurrences[word] = count + 1

    delete = []
    for key, value in occurrences.items():
        if value < min_occurrence:
            delete.append(key)
    for key in delete:
        occurrences.__delitem__(key)

    return set(occurrences.keys())


def count_review_frequencies(reviews, labels, unique_words):
    pos_count, neg_count = dict.fromkeys(unique_words, 0), dict.fromkeys(unique_words, 0)

    for review, label in zip(reviews, labels):
        for word in set(clean_and_split(review)):
            if word in unique_words:
                if label == "positive":
                    pos_count[word] = pos_count.get(word) + 1
                elif label == "negative":
                    neg_count[word] = neg_count.get(word) + 1

    return pos_count, neg_count


def likelihood_present(pos_freq_dict, neg_freq_dict, training_labels):
    alpha = 1       # Smoothing factor
    pos_prob_dict, neg_prob_dict = {}, {}

    pos_count, neg_count = training_labels.count("positive"), training_labels.count("negative")
    total = pos_count + neg_count
    pos_prior, neg_prior = pos_count / total, neg_count / total

    # positive_occurrences and negative_occurrences are the same length
    for (key_pos, value_pos), (key_neg, value_neg) in zip(pos_freq_dict.items(), neg_freq_dict.items()):
        pos_prob_dict[key_pos] = (value_pos + alpha) / (pos_count + (alpha * 2))
        neg_prob_dict[key_neg] = (value_neg + alpha) / (neg_count + (alpha * 2))

    return pos_prob_dict, neg_prob_dict, pos_prior, neg_prior


def predict_labels(evaluation_data, pos_prob_dict, neg_prob_dict, pos_prior, neg_prior):
    prediction = []
    for entry in evaluation_data:
        score_positive, score_negative = 1, 1

        for word in clean_and_split(entry):
            if word in pos_prob_dict:
                score_positive += log(pos_prob_dict[word])
            if word in neg_prob_dict:
                score_negative += log(neg_prob_dict[word])

        if score_positive - score_negative > log(pos_prior) - log(neg_prior):
            prediction.append("positive")
        else:
            prediction.append("negative")
    return prediction


def check_accuracy(actual_labels, predicted_labels):
    confusion = metrics.confusion_matrix(actual_labels, predicted_labels)
    accuracy = round(metrics.accuracy_score(actual_labels, predicted_labels) * 100, 2)
    print("Confusion Matrix: ")
    print(confusion)
    print("Accuracy: " + str(accuracy) + "%")
    return accuracy


def evaluate_k_fold(data, labels, min_length, min_occurrence, splits):
    results = []
    data_series = pd.Series(data)
    labels_series = pd.Series(labels)
    k_fold = model_selection.KFold(n_splits=splits, shuffle=True)
    for train_indexes, test_indexes in k_fold.split(data_series):
        train_data, train_labels = list(data_series.iloc[train_indexes]), list(labels_series.iloc[train_indexes])
        test_data, test_labels = list(data_series.iloc[test_indexes]), list(labels_series.iloc[test_indexes])

        unique_words = get_unique_words(train_data, min_length, min_occurrence)
        pos_freq_dict, neg_freq_dict = count_review_frequencies(train_data, train_labels, unique_words)
        pos_prob_dict, neg_prob_dict, pos_prior, neg_prior = likelihood_present(pos_freq_dict, neg_freq_dict,
                                                                                train_labels)
        predicted_labels = predict_labels(test_data, pos_prob_dict, neg_prob_dict, pos_prior, neg_prior)
        results.append(check_accuracy(test_labels, predicted_labels))

    print("Best accuracy: " + str(np.max(results)) + "%")
    print("Average accuracy: " + str(round(np.average(results))) + "%")
    print("Mean accuracy: " + str(round(float(np.mean(results)), 2)) + "%")


main()
