import pandas as pd
from glob import glob
import csv
import math
import random

def load_File_Data(fileName):
    lines = csv.reader(open(fileName, "rt"))
    data_set = list(lines)
    line_count=0
    for i in range(len(data_set)):
        if line_count != 0:
            data_set[i] = [float(x) for x in data_set[i]]

    return data_set

# tinh toan gia tri trung binh cua moi thuoc tinh
def mean(numbers):
    return sum(numbers) / float(len(numbers))

#tinh do giam cua co phieu
def reduction(number1,number2):
    return number2-number1

# Tinh toan do lech chuan cho tung thuoc tinh
def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)

    return math.sqrt(variance)

# Chuyen ve cap du lieu  (Gia tri trung binh , do lech chuan)
def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries

def summarize_by_class(dataset):
    separated = separate_data(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)

    return summaries

# Phan chia tap du lieu theo class
def separate_data(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    return separated

# Tinh toan xac suat theo phan phoi Gause cua bien lien tuc the hien cac chi so
def calculate_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Tinh xac suat cho moi chi so theo class
def calculate_class_prob(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)

    return probabilities

# Phan chia tap du lieu thanh training va testing. Co the dung train_test_split
def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))

    return [trainSet, copy]

# Du doan vector thuoc phan lop nao
def predict(summaries, inputVector):
    probabilities = calculate_class_prob(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel

# Du doan tap du lieu testing thuoc vao phan lop nao
def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions

# Tinh toan do chinh xac cua phan lop
def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(testSet))) * 100.0

def main():
    fileNames = glob("../data/*.csv")
    load_File_Data(fileNames)

if __name__ == "__main__":
    main()