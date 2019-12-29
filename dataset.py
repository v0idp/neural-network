from PIL import Image
import numpy as np
import os
import csv
import random

class Dataset:
    def __init__(self, split):
        self.directories = []
        self.train_data = []
        self.test_data = []
        self.split = split
        for ds in os.listdir('dataset'):
            self.directories.append(ds)
        counter = 0
        for d in self.directories:
            for file in os.listdir('dataset/{0}'.format(d)):
                img = Image.open('dataset/{0}/{1}'.format(d, file)).convert('L')
                tarr = np.array(img).flatten().tolist()
                tarr = [255 - x for x in tarr]
                tarr.insert(0, counter)
                self.train_data.append(tarr)
            counter += 1
        random.shuffle(self.train_data)
        self.test_data = self.train_data[int(len(self.train_data)*self.split):]
        del self.train_data[int(len(self.train_data)*self.split):]

    def convert_to_csv(self):
        wtr = csv.writer(open('mnist_train.csv', 'w'), delimiter=',', lineterminator='\n')
        for data in self.train_data:
            wtr.writerow(data)
        wtr = csv.writer(open('mnist_test.csv', 'w'), delimiter=',', lineterminator='\n')
        for data in self.test_data:
            for d in data:
                wtr.writerow(d)

    def load_data(self):
        return self.train_data, self.test_data
