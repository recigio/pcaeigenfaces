from person import Person
import os
import sys
import numpy as np
import random
import cv2 as cv
    
class ImageHandler():

    def __init__(self, path, train_percentage):    
        self.path = path
        self.train_percentage = train_percentage * 10
        
    def get_data(self, file_name):
    
        #le imagem e converte para cinza
        image = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
        resized = cv.resize(image, (80, 80))

        #transforma para vetor e coluna
        matriz = resized.T.reshape((1, resized.shape[1] * resized.shape[0]))

        # Converte de 8 bits sem sinal, para 64 bits com sinal, preserva 1 canal apenas.
        data = np.float64(matriz)
        return data

    def load_dataset(self, number_samples_label):
    
        #carrega dataset
        train = []
        test = []
        for root, _, files in os.walk(self.path):
            images = [os.path.join(root, file) for file in files if file.endswith(".jpg")]

        dataSet = []
        for image in images:
            data_part = image[image.rfind("\\") + 1 : image.rfind(".jpg")]
            data = data_part.split("_")
            dataFile = self.get_data(image)
            person = Person(int(data[0]), int(data[1]), dataFile)
            dataSet.append(person)
  
        dataSet.sort(key=lambda person: person.id)

        index = 0
        while index < len(dataSet):
            samples = dataSet[index: index + number_samples_label]

            while len(samples) > self.train_percentage:
                i = random.randint(0, len(samples) - 1)
                test.append(samples.pop(i))

            if self.train_percentage == number_samples_label:
                test.extend(samples)    

            train.extend(samples)
            index += number_samples_label

        return (train, test)     
        