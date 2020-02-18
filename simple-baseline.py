"""
Simple Baseline
Contains 3 Algorithms to Predict Recidivism & a Visualization of their Results:
    - Random guessing
    - Stereotyped assumption
    - COMPAS algorithm (results visualization & analysis only)

-------------------------------------------------
TO RUN
    1: change GROUND_TRUTH_NAME to the ground truth value in your input dataset
    2: run the command: python3 simple-baseline.py __filenamehere__

-------------------------------------------------
"""

import argparse
from collections import defaultdict
import scipy
import copy
import matplotlib.pyplot as plt
import math
from random import randint
import json

DECILE_SCORE_NAME = 'decile_score'
#GROUND_TRUTH_NAME = 'two_year_recid'
GROUND_TRUTH_NAME = 'is_recid'
DID_RECIDIVISE_NAME = '1'
NO_RECIDIVISE_NAME = '0'
NUMERIC_ISH_FEATURESP_1 = ['DateOfBirth', 'Person_ID', 'AssessmentID', 'Case_ID', 'Screening_Date', 'RawScore', 'DecileScore']
NUMERIC_ISH_FEATURES = ['id', 'age', 'decile_score']
JSON_FILE_PATH = "../Fairness/simpleBaselineData.json"

"""
-------------------------------------------------
    Section 0: MAIN
-------------------------------------------------
"""
def main():
    args = parse_args()
    features, features_with_values, all_people = load_data(args.filename)
    display_all_methods(all_people)


"""
-------------------------------------------------
Section 1: Loading the data
-------------------------------------------------
"""
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("filename", help="Filepath to the data in a csv format")
    args = p.parse_args()
    return args

### Input: csv file
### Output:
##      - features: a list of all the features in the data (col headers)
##      - features_w_values: default dict
##             {'sex': {'Female', 'Male'}, 'race': {'Hispanic',...},...}
##      - all_people: a list of dictionaries (each dict represents a person)
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
                elif j < len(features):
                    if features[j] not in NUMERIC_ISH_FEATURES  and split_line[j] not in features_with_values[features[j]]:
                        features_with_values[features[j]].add(split_line[j])
                    if features[j] == GROUND_TRUTH_NAME and split_line[j] == '':
                        person[features[j]] = 0
                    else:
                        person[features[j]] = split_line[j]
                    all_people.append(person)
            i += 1
        return features, features_with_values, all_people


"""
-------------------------------------------------
Section 2: Simple Random Baseline
-------------------------------------------------
"""
# Input: a list of dictionaries (where each dict represents a person)
# Output: a list of guesses (0 or 1), a list of decile scores (1 or 10)
def random_baseline(data):
    total_recid = 0 # Total number of people who have recidivated
    for person in data:
        total_recid += int(person[GROUND_TRUTH_NAME])

    rate_recid = total_recid/len(data)

    guesses = []
    decile = []
    count_less = 0
    for i in range(len(data)):
        rand_num = randint(0,1) # use a random number to guess 0 or 1
        if rand_num == 0:
            guesses.append(0)
            decile.append(1)
            count_less += 1
        else:
            guesses.append(1)
            decile.append(10)
    return guesses, decile

"""
-------------------------------------------------
Section 3: Stereotyped Condition
-------------------------------------------------
"""
# Assume all men under 34 will recidivate, and everyone else will not recidivate
# Input: a dictionary representing a person
# Output: a boolean - True (if person < 34), False (o/w)
def foolish_condition(person):
    if int(person['age']) < 34 and person['sex'] == 'Male':
        return True
    return False

def transform_decile_true(score):
    return 10

def transform_decile_false(score):
    return 1

# Input: old_data (a list of people rep. by dictionaries), condition (a function),
#        transformation_false (a function)
# Output: a list of people rep. by dictionaries
# More info: Goes through dataset, if a person meets the "foolish-condition" then
#           assign a decile score of 10 (ie will recidivate), o/w assign score 1
def transform_score(old_data, condition, transform, transform_false,
                    true_param=None, false_param=None):
    data = copy.deepcopy(old_data)
    for person in data:
        if condition(person):
            if true_param != None:
                person['decile_score'] = transform(person['decile_score'], true_param)
            else:
                person['decile_score'] = transform(person['decile_score'])
        else:
            if false_param != None:
                person['decile_score'] = transfor_false(person['decile_score'], false_param)
            else:
                person['decile_score'] = transform_false(person['decile_score'])
    return data

"""
-------------------------------------------------
Section 4: Calculating Metrics of Evaluation
-------------------------------------------------
"""
# Input: list of guesses (0,1) and a list of dicts (where each dict rep a person), ie all data
# Output: integer between 0-1 representing what % of the guesses were correct
def get_accuracy(guesses, correct):
    total_correct = 0
    num_zero = 0
    num_one = 0
    for i in range(len(guesses)):
        if guesses[i] == int(correct[i][GROUND_TRUTH_NAME]):
            total_correct += 1
    return total_correct/len(guesses)

# Input: a list of dictionaries (where each dict represents a person)
# Output: integer between 0-1 representing what % of the guesses were correct, guesses list (0 or 1)
# More info: COMPAS assigns a recividism likelihood score of 1-10 (least to most likely)
#            this function assumes all people with scores 1-5 will not recidivate (0)
#            & that all people with scores 6-10 will recidivate (1)
def get_acc_compas(data):
    guesses = []
    for person in data:
      if int(person['decile_score']) > 5:
          guesses.append(1)
      else:
          guesses.append(0)
    return guesses, get_accuracy(guesses, data)

# Input: a list of ints (0-10), a list of dictionaries (where each dict represents a person)
# Output: a list of ints (0-10) represnting how "far" from correct the COMPAS guesses were
# More info: If for personA COMPAS guessed "4", and the ground truth value was 1,
#            then the distance from correct for personA is 6
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

# Input: a list of ints (0-10), a list of dictionaries (where each dict represents a person)
# Output: a float, representing the mean of all the "distances from correct"
def get_mean_distance(guesses, data):
    distances = get_distances_from_correct(guesses, data)
    return sum(distances)/len(distances)

# Input: a list of ints (0-10), a list of dictionaries (where each dict represents a person)
# Output: a float, representing the mean of all the "distances from correct"
def get_mean_distance_compas(data):
    guesses = [person[DECILE_SCORE_NAME] for person in data]
    distances = get_distances_from_correct(guesses, data)
    return sum(distances)/len(distances)

# Input: a list of people (rep by dictionaries), ie the entire dataset
# Output: a list of ints - [true pos rate, false pos rate, true neg rate, false neg rate]
# More info: "get_positiverates_negativerates_compas()"
def get_pr_nr_compas(all_people):
    #return get_pr_nr([person[DECILE_SCORE_NAME] for person in all_people], [person[GROUND_TRUTH_NAME] for person in all_people])
    return get_pr_nr([person[DECILE_SCORE_NAME] for person in all_people], all_people)

# Input: a list of predicted values (0 | 1), a list of actual truth values (0 | 1)
# Output: a list of ints - [true pos rate, false pos rate, true neg rate, false neg rate]
# More info: "get_positiverates_negativerates()"
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
            if actual[i][GROUND_TRUTH_NAME] == NO_RECIDIVISE_NAME:
                fp += 1
        elif int(predicted[i]) < 5:
            n += 1
            if actual[i][GROUND_TRUTH_NAME] == NO_RECIDIVISE_NAME:
                tn += 1
    tp = p - fp
    fn = n - tn
    if fp+tn == 0:
        return 0
    else:
        return [tp/(tp+fn), fp/(fp+tn), tn/(tn+fp), fn/(fn+tp)]

"""
-------------------------------------------------
Section 5: Plotting the data
-------------------------------------------------
"""

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

# Input: a list [true pos rate, false pos rate, true neg rate, false neg rate],
#        an int between 0-1 representing accuracy, a string (for title of figure)
# Output: a bar graph showing: true pos rate, false pos rate, true neg rate,
#           false neg rate, accuracy
def plot_pr_nr_acc(pr_nr, accuracy, fig_title):
    y = pr_nr + [accuracy]
    x = [0, 1, 2, 3, 4]
    plt.bar(x,y, color=["lightblue","lightblue","lightblue","lightblue","darkred"])
    plt.margins(0.1, tight=True)

    # create labels for above each bar
    labels = []
    for val in y:
        labels.append(float(int(val*1000))/1000) # round all y vals to hundreds place
    # set labels in place
    for i in range(len(y)):
        plt.text(x=x[i]-0.06, y=y[i]+0.005, s=labels[i])

    plt.xticks(x, ('TPR', 'FPR', 'TNR', 'FNR', "Accuracy"))

    fig = plt.gcf()
    fig.canvas.set_window_title(fig_title)
    plt.title(fig_title)
    plt.show()

# Input: a list of ints (0-1) representing the accuracy of various algorithms,
#        a list of strings to be used as labels for the values in the list of accuracies
# Output: a bar graph comparing the accuracies of various algorithms
def plot_all_accuracy(accuracies, accuracy_labels):
    y = accuracies
    x = []
    for i in range(0,len(accuracies)):
        x.append(i)
    plt.bar(x,y, color="lightblue")
    plt.margins(0.1, tight=True)

    # create labels for above each bar
    labels = []
    for val in y:
        labels.append(float(int(val*1000))/1000) # round all y vals to hundreds place
    # set labels in place
    for i in range(len(y)):
        plt.text(x=x[i]-0.06, y=y[i]+0.005, s=labels[i])

    if len(accuracies) == len(accuracy_labels):
        plt.xticks(x, accuracy_labels)

    fig = plt.gcf()
    fig.canvas.set_window_title("Accuracy Comparison")
    plt.title("Comparison of Accuracy Across Algorithms")
    plt.show()

'''
-------------------------------------------------
Section 6: Create CSV/JSON files to export algorithm results
-------------------------------------------------
'''
# borrowed from split-cat-num.py
def create_filestring(data):
    # creates a string to write to a file based on the passed list
    string = ''
    for person in data:
        for attribute in person:
            string += str(attribute)
            string += ","
        string = string[:-1]
        string += "\n"
    return string

# borrowed from split-cat-num.py
def create_file(filename, data,create_filestring=True):
    # writes a csv file in `filename` based containing `data`
    if create_filestring:
        string = create_filestring(data)
    else:
        string = data
    with open(filename, 'w') as file:
        file.write(string)

def getTruthGuessString(guesses, all_data):
    string = "predicted,truth\n"
    for i in range(len(guesses)):
        newLine = str(guesses[i]) + "," + str(all_data[i][GROUND_TRUTH_NAME]) + "\n"
        string += newLine
    return string


'''
-------------------------------------------------
Section 7: Calling All Algorithms
-------------------------------------------------
'''
# Provides results of random baseline, foolish assumption, and COMPAS
# via printed float values and pop-up bar graphs
def display_all_methods(all_people):
    print("-------------------------------------------------")

    # RANDOM BASELINE
    guesses, decile = random_baseline(all_people)
    acc = get_accuracy(guesses, all_people)
    dist = get_mean_distance(decile, all_people)
    print("\nRandom baseline accuracy:", acc)
    print("Random baseline mean distance:", dist)
    #plot_dist(decile, all_people)
    plot_pr_nr_acc(get_pr_nr(decile, all_people), acc, "Random Baseline Results")
    create_file("randomBaselineResults.csv",getTruthGuessString(guesses, all_people), False)

    # COMPAS
    compas_guesses, acc_compas = get_acc_compas(all_people)
    dist_compas = get_mean_distance_compas(all_people)
    print("\nCOMPAS accuracy:", acc_compas)
    print("COMPAS mean distance:", dist_compas)
    #plot_dist_compas(all_people)
    plot_pr_nr_acc(get_pr_nr_compas(all_people), acc_compas, "COMPAS Results")

    # FOOLISH ASSUMPTION
    transformed_data = transform_score(all_people, foolish_condition, transform_decile_true, transform_decile_false)
    foolish_guesses, acc_foolish = get_acc_compas(transformed_data)
    dist_foolish = get_mean_distance_compas(transformed_data)
    print("\nFoolish Transformation accuracy:", acc_foolish)
    print("Foolish Transformation mean distance:", dist_foolish)
    #plot_dist_compas(transformed_data)
    plot_pr_nr_acc(get_pr_nr_compas(transformed_data), acc_foolish, "Stereotyped Condition Results")
    create_file("foolishConditionResults.csv",getTruthGuessString(foolish_guesses, all_people), False)

    # Create a bar chart comparing the accuracies of the 3 algorithms
    all_accs = [acc, acc_compas, acc_foolish]
    acc_labels = ["Random Basline", "COMPAS", "Stereotyped Condition"]
    plot_all_accuracy(all_accs, acc_labels)

    # Create a json file with the results of the 3 algorithms
    data_guesses_dict = {"people":all_people, "random":guesses,
                         "foolish":foolish_guesses, "compas":compas_guesses}
    data_guesses_str = str(data_guesses_dict)

    with open(JSON_FILE_PATH, 'w') as file:
        json.dump(data_guesses_dict, file)

    print("-------------------------------------------------")

"""
-------------------------------------------------
Section 8: Other
-------------------------------------------------
"""
# Input: score (int), translate (int, default val 2)
# Output: an int -- 10 or the sum of score and translate (if sum < 10)
# More info: Ensures that a decile score is no greater than 10
def translate_score(score, translation=2):
    if int(score) + translation >= 10:
        return 10
    return int(score) + translation

# Input: score (int), translate (int, default val 2)
# Output: an int -- 1 or the difference of score and translate (if diff > 1)
# More info: Ensures that a decile score is no less than 1
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
            guesses, acc = get_acc_compas(transformed_data)
            inner_acc.append(acc)
            inner_dists.append(get_mean_distance_compas(transformed_data))
        accs.append(inner_acc)
        dists.append(inner_dists)
    best = 0.58
    for i in range(len(accs)):
        for j in range(len(accs[i])):
            if accs[i][j] > best:
                print("Accuracy of", accs[i][j], "acheived at true-case translation of", i, "and false-case translation of", j)

"""
-------------------------------------------------
Section 9: Discuss with Group (TODO)
-------------------------------------------------
"""
# Issue: uses the recidivism rate in the data to
#   Isn't this a problem, becuase we're using the "answer" to calculate our guess
# Should be re-written to guess randomly everytime, and the accuracy should match
# the recidivism rate
def random_baseline_old(data):
    total = 0
    for person in data:
        total += int(person[GROUND_TRUTH_NAME])

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





if __name__ == "__main__":
    main()
