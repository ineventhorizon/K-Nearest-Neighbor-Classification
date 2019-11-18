import numpy as np
import cv2
import glob
import math


#Reads given path from the files, resizes all images to 128x128 and transforms it into a
#vector. Returns np array with shape NumberofDatax16384
def getInput(path, Class = 'default', train = False):
    print(f'Reading {path} data')
    data = np.array([cv2.resize(cv2.imread(file, 0), (128, 128)).flatten() for file in glob.glob(f'{path}/*.jpg')], dtype=int)
    if train:
        cls = np.full((data.shape[0], 1), Class, dtype=object)
        return data, cls
    else:
        return data


#Calculates Euclidean Distance for given x and y rows.
#Returns calculated distance.
def euclideanDistance(x, y):
    row = x-y
    row = np.square(row, dtype=float)
    return math.sqrt(np.sum(row))


#Function for K-Nearest-Neighbors algorithm.
#Params x_train: Images, y_train: Classes, test: Test data, k: Nearest k items
#Returns predicted Classes for test data
def KNN(x_train, y_train, test, k):
    print("Starting to calculate.")
    #Distance matrix for test data.
    distance = np.zeros((test.shape[0], x_train.shape[0]), dtype=float)
    #Calculates distance of each test data with respect to the training data.
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            distance[i][j] = euclideanDistance(test[i], x_train[j])
    #To return the results created an array with shape of (NumberOfTestData,1)
    prediction = np.empty((test.shape[0], 1),  dtype=object)

    #Sorts calculated distances in ascending order based on distance values
    #Gets top k row from the sorted array and get the most frequent class of these rows.
    #If there are two or more items in the result array it will
    #decrases k by one until there is only one matching class.
    for test_item in range(test.shape[0]):
        p = distance[test_item].argsort()
        first = y_train[p]
        unique_elements, counts_elements = np.unique(first[:k], return_counts=True)
        result = np.where(counts_elements == np.amax(counts_elements))
        sub = 1
        while unique_elements[result].shape[0] != 1 or sub == k-1:
            unique_elements, counts_elements = np.unique(first[:k-sub], return_counts=True)
            result = np.where(counts_elements == np.amax(counts_elements))
        prediction[test_item] = unique_elements[result]
    print("Calculation finished.")
    print("Predicted classes for test data is: \n", prediction)
    return prediction


butterfly_train, butterfly_class = getInput('train/butterfly', 'Butterfly', True)
chair_train, chair_class = getInput('train/chair', 'Chair', True)
laptop_train, laptop_class = getInput('train/laptop', 'Laptop', True)
test_data = getInput('test')
train_x = np.concatenate((butterfly_train, chair_train, laptop_train))
train_y = np.concatenate((butterfly_class, chair_class, laptop_class))
prediction = KNN(train_x, train_y, test_data, 7)

