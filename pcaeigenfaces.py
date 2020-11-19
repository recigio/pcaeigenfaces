### open cv import
import cv2 as cv
import numpy as np
import sys

## classes do projeto
from dataset_handler import ImageHandler

#parametros
train = []
test = []
path= './ORL'
train_percent = 0.7
min_comps = 10
max_comps = 21

#Load do dataset
imageHandler = ImageHandler(path, train_percent)
train, test = imageHandler.load_dataset(10)


print(f"Porcentagem de treinamento: {train_percent * 100}%")
print(f"Quantidade imagens de treino: {len(train)}")
print(f"Quantidade imagens de teste: {len(test)}")
print("-------------------")

for number_components in range(min_comps, max_comps):
    
    model = cv.face.EigenFaceRecognizer_create(number_components)
    src = []
    labels = []
    
    for person in train:
        src.append(person.data)
        labels.append(person.label)

    model.train(src, np.asarray(labels))

    max_distance = sys.float_info.max
    min_distance = sys.float_info.min
    
    mean_distance = 0
    corrects = 0
    true_positives = 0
    
    for person in test:
        label, confidence = model.predict(person.data)
        
        if confidence < min_distance:
            min_distance = confidence

        if confidence > max_distance:
            max_distance = confidence
            
        mean_distance += confidence
            
        if person.label == label:
            corrects += 1
            true_positives += 1

    accuracy = true_positives / len(test) * 100
    mean_distance /= len(test)
    
    print(f"{number_components} componentes principais")
    print(f"Distância mínima: {min_distance}")
    print(f"Distância máxima: {max_distance}")
    print(f"Distância média: {mean_distance}")
    print(f"Acurracia: {accuracy}")
    print(f"Acertos: {corrects}")
    print("-------------------")