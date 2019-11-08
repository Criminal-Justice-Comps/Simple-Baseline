""" A first try at a simple random baseline.

TO RUN: python3 read_in_prelim.py __filename__
"""


import argparse
from collections import defaultdict
import scipy
import copy
import matplotlib.pyplot as plt
import math

DECILE_SCORE_NAME = 'decile_score'
GROUND_TRUTH_NAME = 'two_year_recid'
DID_RECIDIVISE_NAME = '1'
NO_RECIDIVISE_NAME = '0'
NUMERIC_ISH_FEATURESP_1 = ['DateOfBirth', 'Person_ID', 'AssessmentID', 'Case_ID', 'Screening_Date', 'RawScore', 'DecileScore']
NUMERIC_ISH_FEATURES = ['id', 'age', 'decile_score']

def main():
    args = parse_args()
    features, features_with_values, all_people = load_data(args.filename)
    display_all_methods(all_people)


def plot_dist(guesses, data):
    distances = get_distances_from_correct(guesses, data)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    histogram = plt.hist(distances, bins, density=True)
    plt.show()

def plot_dist_compas(data):
    distances = get_distances_from_correct([person[DECILE_SCORE_NAME] for person in data], data)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    histogram = plt.hist(distances, bins, density=True)
    plt.show()

def plot_pr_nr_compas(all_people):
    y = get_pr_nr_compas(all_people)
    x = [0, 1, 2, 3]
    figure = plt.bar(x, y)
    plt.show()

def plot_pr_nr(guesses, actual):
    y = get_pr_nr(guesses, actual)
    x = [0, 1, 2, 3]
    figure = plt.bar(x, y)
    plt.show()

def display_all_methods(all_people):
    guesses, decile = random_baseline(all_people)
    acc = get_accuracy(guesses, all_people)
    dist = get_mean_distance(decile, all_people)
    print("Random baseline accuracy is:", acc)
    print("Random baseline mean distance is:", dist)
    #plot_dist(decile, all_people)
    plot_pr_nr(decile, all_people)

    acc_compas = get_acc_compas(all_people)
    dist_compas = get_mean_distance_compas(all_people)
    print("COMPAS accuracy is:", acc_compas)
    print("COMPAS mean distance is:", dist_compas)
    #plot_dist_compas(all_people)
    plot_pr_nr_compas(all_people)

    transformed_data = transform_score(all_people, foolish_condition, foolish_transformation, foolish_transformation_false)
    acc_foolish = get_acc_compas(transformed_data)
    dist_foolish = get_mean_distance_compas(transformed_data)
    print("Foolish Transformation accuracy is:", acc_foolish)
    print("Foolish Transformation mean distance is:", dist_foolish)
    #plot_dist_compas(transformed_data)
    plot_pr_nr_compas(transformed_data)


    #translation_grid_search(all_people)
    """transformed_data_2 = transform_score(all_people, foolish_condition, translate_score, translate_score_false)
    acc_foolish_2 = get_acc_compas(transformed_data_2)
    dist_foolish_2 = get_mean_distance_compas(transformed_data_2)
    print("Foolish Transformation #2 has accuracy:", acc_foolish_2)
    print("Foolish Transformation #2 has mean distance:", dist_foolish_2)
    plot_dist_compas(transformed_data_2)

    for i in range(30):
        print(transformed_data[i])
    print("-----"*40)
    for i in range(30):
        print(all_people[i])"""
    #acc = get_acc_compas(transformed_data)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("filename", help="Filepath to the data in a csv format")
    args = p.parse_args()
    return args



def load_data(filepath):
    with open(filepath) as f:
        i = 0
        features = []
        features_with_values = defaultdict(set)
        all_people = []
        for line in f:
            person = {}
            split_line = line.split(",")
            split_line[-1] = split_line[-1][:-1]
            for j in range(len(split_line)):
                if i == 0:
                    features.append(split_line[j])
                else:
                    if features[j] not in NUMERIC_ISH_FEATURES  and split_line[j] not in features_with_values[features[j]]:
                        features_with_values[features[j]].add(split_line[j])
                    if features[j] == 'two_year_recid' and split_line[j] == '':
                        person[features[j]] = 0
                    else:
                        person[features[j]] = split_line[j]
                    all_people.append(person)
            i += 1
        return features, features_with_values, all_people

def random_baseline(data):
    total = 0
    for person in data:
        total += int(person['two_year_recid'])
    rate_recid = total/len(data)
    guesses = []
    decile = []
    for i in range(len(data)):
        guess = scipy.random.ranf()
        if guess <= rate_recid:
            guesses.append(1)
            decile.append(10)
        else:
            guesses.append(0)
            decile.append(1)
    return guesses, decile

def get_accuracy(guesses, correct):
    total_correct = 0
    num_zero = 0
    num_one = 0
    for i in range(len(guesses)):
        if guesses[i] == int(correct[i]['two_year_recid']):
            total_correct += 1
    return total_correct/len(guesses)

def get_acc_compas(data):
    guesses = []
    for person in data:
      if int(person['decile_score']) > 5:
          guesses.append(1)
      else:
          guesses.append(0)
    return get_accuracy(guesses, data)

def get_distances_from_correct(guesses, data):
    distances = []
    for i in range(len(data)):
        person = data[i]
        guess = int(guesses[i])
        if person[GROUND_TRUTH_NAME] == DID_RECIDIVISE_NAME:
            distance = 10 - guess
            distances.append(distance)
        else:
            distance = guess - 1
            distances.append(distance)
    if -1 in distances:
        print("Impossible score in distances")
        exit(0)
    return distances

def get_mean_distance(guesses, data):
    distances = get_distances_from_correct(guesses, data)
    return sum(distances)/len(distances)

def get_mean_distance_compas(data):
    guesses = [person[DECILE_SCORE_NAME] for person in data]
    distances = get_distances_from_correct(guesses, data)
    return sum(distances)/len(distances)

def transform_score(old_data, condition, transformation, transformation_false, true_param=None, false_param=None):
    data = copy.deepcopy(old_data)
    for person in data:
        if condition(person):
            if true_param != None:
                person['decile_score'] = transformation(person['decile_score'], true_param)
            else:
                person['decile_score'] = transformation(person['decile_score'])
        else:
            if false_param != None:
                person['decile_score'] = transformation_false(person['decile_score'], false_param)
            else:
                person['decile_score'] = transformation_false(person['decile_score'])
    return data

def foolish_condition(person):
    if int(person['age']) < 34 and person['sex'] == 'Male':
        return True
    return False

def foolish_transformation(score):
    return 10

def foolish_transformation_false(score):
    return 1

def translate_score(score, translation=2):
    if int(score) + translation >= 10:
        return 10
    return int(score) + translation

def translate_score_false(score, translation=2):
    if int(score) - translation <= 1:
        return 1
    return int(score) - translation

def translation_grid_search(all_people, start=0, stop=10, step=1):
    accs = []
    dists = []
    for i in range(start, stop + 1, step):
        inner_acc = []
        inner_dists = []
        for j in range(start, stop + 1, step):
            transformed_data = transform_score(all_people, foolish_condition, translate_score, translate_score_false, i, j)
            inner_acc.append(get_acc_compas(transformed_data))
            inner_dists.append(get_mean_distance_compas(transformed_data))
        accs.append(inner_acc)
        dists.append(inner_dists)
    best = 0.58
    for i in range(len(accs)):
        for j in range(len(accs[i])):
            if accs[i][j] > best:
                print("Accuracy of", accs[i][j], "acheived at true-case translation of", i, "and false-case translation of", j)

def get_pr_nr_compas(all_people):
    return get_pr_nr([person[DECILE_SCORE_NAME] for person in all_people], [person[GROUND_TRUTH_NAME] for person in all_people])

def get_pr_nr(predicted, actual):
    fp = 0
    tn = 0
    p = 0
    n = 0
    total = len(predicted)
    if len(predicted) != len(actual): return 0
    for i in range(0,len(predicted)):
        if int(predicted[i]) > 5:
            p += 1
            if actual[i] == NO_RECIDIVISE_NAME:
                fp += 1
        elif int(predicted[i]) < 5:
            n += 1
            if actual[i] == NO_RECIDIVISE_NAME:
                tn += 1
    tp = p - fp
    fn = n - tn
    if fp+tn == 0:
        return 0
    else:
        return [tp/(tp+fn), fp/(fp+tn), tn/(tn+fp), fn/(fn+tp)]




if __name__ == "__main__":
    main()
