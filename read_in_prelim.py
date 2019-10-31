""" A first try at a simple random baseline.

TO RUN: python3 read_in_prelim.py __filename__
"""


import argparse
from collections import defaultdict
import scipy
import copy

NUMERIC_ISH_FEATURESP_1 = ['DateOfBirth', 'Person_ID', 'AssessmentID', 'Case_ID', 'Screening_Date', 'RawScore', 'DecileScore']
NUMERIC_ISH_FEATURES = ['id', 'age', 'decile_score']

def main():
    args = parse_args()
    features, features_with_values, all_people = load_data(args.filename)
    guesses = random_baseline(all_people)
    acc = get_accuracy(guesses, all_people)
    print("Random baseline accuracy is:", acc)
    acc_compas = get_acc_compas(all_people)
    print("COMPAS accuracy is:", acc_compas)
    transformed_data = transform_score(all_people, foolish_condition, foolish_transformation, foolish_transformation_false)
    acc_foolish = get_acc_compas(transformed_data)
    print("Foolish Transformation accuracy is:", acc_foolish)
    transformed_data_2 = transform_score(all_people, foolish_condition, foolish_transformation_2, foolish_transformation_false_2)
    acc_foolish_2 = get_acc_compas(transformed_data_2)
    print("Foolish Transformation #2 has accuracy is:", acc_foolish_2)
    """for i in range(30):
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
    for i in range(len(data)):
        guess = scipy.random.ranf()
        if guess <= rate_recid:
            guesses.append(1)
        else:
            guesses.append(0)
    return guesses

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


def transform_score(old_data, condition, transformation, transformation_false):
    data = copy.deepcopy(old_data)
    for person in data:
        if condition(person):
            person['decile_score'] = transformation(person['decile_score'])
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
    return 0

def foolish_transformation_2(score):
    if int(score) + 1 >= 10:
        return 10
    return int(score) + 1

def foolish_transformation_false_2(score):
    if int(score) - 1 <= 0:
        return 0
    return int(score) - 1

if __name__ == "__main__":
    main()
