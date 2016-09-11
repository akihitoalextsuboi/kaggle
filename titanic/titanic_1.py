import csv as csv
import numpy as np
import sys as sys
from os.path import dirname
# sys.path.append(dirname(__file__))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

print 'started'
csv_file_object = csv.reader(open('./csvs/train.csv', 'rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
    data.append(row)

data = np.array(data)

test_file = open('./csvs/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header_2 = test_file_object.next()
prediction_file = open("./csvs/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object: 
    if row[3] == 'female':  
        prediction_file_object.writerow([row[0],'1'])
    else:
        prediction_file_object.writerow([row[0],'0'])

test_file.close()
prediction_file.close()
print 'finished'
