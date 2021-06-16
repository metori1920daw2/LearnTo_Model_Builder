from tensorflow.keras.datasets import mnist
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime

def az_dataset(datasetPath):
    data = []
    labels = []
    for row in open(datasetPath):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    return (data, labels)

def mnist_dataset():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    return (data, labels)

def load_dataset(datasetPath):
    print("Loading data!")
    (azData, azLabels) = az_dataset(datasetPath)
    (digitsData, digitsLabels) = mnist_dataset()
    """azLabels += 10
    data = np.vstack([azData, digitsData])
    labels = np.hstack([azLabels, digitsLabels])
    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data = np.repeat(data, 3, axis=-1)
    data /= 255.0"""
    azLabels += 10
    data = np.vstack([azData, digitsData])
    labels = np.hstack([azLabels, digitsLabels])
    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    le = LabelBinarizer()
    labels = le.fit_transform(labels)
    classTotals = labels.sum(axis=0)
    classWeight = {}
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, stratify=labels, random_state=42)
    return  (trainX, testX, trainY, testY,classWeight)