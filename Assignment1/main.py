import re
import numpy as np
import pandas as pd
from math import log
from collections import defaultdict
from sklearn import metrics, model_selection


def main(min_length, min_occurrence, k_fold):
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

    if not k_fold:
        return check_accuracy(test_labels, predicted_labels)

    check_accuracy(test_labels, predicted_labels)

    print("\nEvaluating k-fold")
    best_values = evaluate_k_fold(train_data, train_labels, min_length, min_occurrence, 10)

    print("\nClassifying evaluation data with optimal dictionaries")
    predicted_labels2 = predict_labels(test_data, best_values[0], best_values[1], best_values[2], best_values[3])
    return check_accuracy(test_labels, predicted_labels2)


# Read in a file and extract training data, test data, training labels and test labels
def read_data(file):
    data = pd.read_excel(file)
    training = data[data["Split"] == "train"]
    evaluation = data[data["Split"] == "test"]

    print("Number of positive training reviews: " + str(len(training[training["Sentiment"] == "positive"])))
    print("Number of negative training reviews: " + str(len(training[training["Sentiment"] == "negative"])))
    print("Number of positive evaluation reviews: " + str(len(evaluation[evaluation["Sentiment"] == "positive"])))
    print("Number of negative evaluation reviews: " + str(len(evaluation[evaluation["Sentiment"] == "negative"])))

    return training["Review"].to_list(), training["Sentiment"].map({"positive": 1, "negative": 0}).to_list(),\
        evaluation["Review"].to_list(), evaluation["Sentiment"].map({"positive": 1, "negative": 0}).to_list()


# Removes non A-Z characters, converts to lowercase and splits text
def clean_and_split(data):
    return re.sub(r'[^\w]', ' ', data).lower().split()


# Cleans a list of reviews and returns a set of unique words that meet the criteria provided
def get_unique_words(data, min_length, min_occurrence) -> set:
    # Count the occurrences of each word if it's length is greater than or equal to min_length
    occurrences = defaultdict(lambda: 0)
    for review in data:
        for word in clean_and_split(review):
            if len(word) >= min_length:
                occurrences[word] += 1

    # Iterate through each word and add if occurrences is greater than min_occurrences
    unique_words = set()
    for key, value in occurrences.items():
        if value >= min_occurrence:
            unique_words.add(key)

    return unique_words


# Count the number of positive and negative reviews each word appears in
def count_review_frequencies(reviews, labels, unique_words):
    # Words will map to 0 if they don't occur. I.e they may occur in positive but not negative
    pos_count, neg_count = dict.fromkeys(unique_words, 0), dict.fromkeys(unique_words, 0)

    for review, label in zip(reviews, labels):
        for word in set(clean_and_split(review)):
            if word in unique_words:
                if label == 1:
                    pos_count[word] += 1
                elif label == 0:
                    neg_count[word] += 1

    return pos_count, neg_count


# Calculate probability dictionaries and priors
def likelihood_present(pos_freq_dict, neg_freq_dict, training_labels):
    alpha = 1       # Smoothing factor
    pos_prob_dict, neg_prob_dict = {}, {}

    pos_count, neg_count = training_labels.count(1), training_labels.count(0)
    total = pos_count + neg_count
    pos_prior, neg_prior = pos_count / total, neg_count / total

    # positive_occurrences and negative_occurrences are the same length
    for (key_pos, value_pos), (key_neg, value_neg) in zip(pos_freq_dict.items(), neg_freq_dict.items()):
        pos_prob_dict[key_pos] = (value_pos + alpha) / (pos_count + (alpha * 2))
        neg_prob_dict[key_neg] = (value_neg + alpha) / (neg_count + (alpha * 2))

    return pos_prob_dict, neg_prob_dict, pos_prior, neg_prior


# Using probability dictionaries label each review as either positive or negative
def predict_labels(evaluation_data, pos_prob_dict, neg_prob_dict, pos_prior, neg_prior):
    prediction = []
    for entry in evaluation_data:
        score_positive, score_negative = 0, 0

        # clean and split review, iterate through each word getting overall scores
        for word in clean_and_split(entry):
            if word in pos_prob_dict:
                score_positive += log(pos_prob_dict[word])
            if word in neg_prob_dict:
                score_negative += log(neg_prob_dict[word])

        # Classify review taking account of ratio of positive to negative reviews
        if score_positive - score_negative > log(neg_prior) - log(pos_prior):
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction


# display info about prediction accuracy
def check_accuracy(actual_labels, predicted_labels):
    confusion = metrics.confusion_matrix(actual_labels, predicted_labels)
    accuracy = round(metrics.accuracy_score(actual_labels, predicted_labels) * 100, 2)
    total = len(predicted_labels)
    print("Confusion matrix: ")
    print(confusion)
    print("True positive: " + str(round((confusion[1][1] / total) * 100, 2)) + "%")
    print("True negative: " + str(round((confusion[0][0] / total) * 100, 2)) + "%")
    print("False positive: " + str(round((confusion[0][1] / total) * 100, 2)) + "%")
    print("False negative: " + str(round((confusion[1][0] / total) * 100, 2)) + "%")
    print("Overall accuracy: " + str(accuracy) + "%")
    return accuracy


# Split data into sections, test and return optimal dictionaries using k-fold cross-validation
def evaluate_k_fold(data, labels, min_length, min_occurrence, splits):
    results = []
    best_accuracy, best_values = 0, ([], [], 0, 0)
    data_series, labels_series = pd.Series(data), pd.Series(labels)
    k_fold = model_selection.KFold(n_splits=splits, shuffle=True)
    i = 0
    for train_indexes, test_indexes in k_fold.split(data_series):
        i += 1
        print("K-fold iteration " + str(i) + " of " + str(splits))
        # Build our training and validation data from indexes
        train_data, train_labels = list(data_series.iloc[train_indexes]), list(labels_series.iloc[train_indexes])
        test_data, test_labels = list(data_series.iloc[test_indexes]), list(labels_series.iloc[test_indexes])

        # Same process as main without as many prints
        unique_words = get_unique_words(train_data, min_length, min_occurrence)
        pos_freq_dict, neg_freq_dict = count_review_frequencies(train_data, train_labels, unique_words)
        pos_prob_dict, neg_prob_dict, pos_prior, neg_prior = likelihood_present(pos_freq_dict, neg_freq_dict,
                                                                                train_labels)
        predicted_labels = predict_labels(test_data, pos_prob_dict, neg_prob_dict, pos_prior, neg_prior)
        accuracy = check_accuracy(test_labels, predicted_labels)
        results.append(accuracy)

        # Record the best dictionaries so far
        if accuracy > best_accuracy:
            best_accuracy, best_values = accuracy, (pos_prob_dict, neg_prob_dict, pos_prior, neg_prior)

    print("Best accuracy: " + str(np.max(results)) + "%")
    print("Average accuracy: " + str(round(np.average(results))) + "%")
    print("Mean accuracy: " + str(round(float(np.mean(results)), 2)) + "%")
    return best_values


# Generate accuracy.txt file with accuracy score for various criteria
def measure_best_accuracy():
    accuracy_file = open("accuracy2.txt", 'w')
    min_occurrences = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

    for o in min_occurrences:
        accuracy_file.write("\t" + str(o))
    accuracy_file.write("\n")

    for l in range(1, 11):
        accuracy_file.write(str(l))
        for o in min_occurrences:
            accuracy_file.write("\t" + str(main(l, o, False)) + "%")
        accuracy_file.write("\n")


main(4, 50, True)
# measure_best_accuracy()
