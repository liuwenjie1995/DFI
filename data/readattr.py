import tensorflow as tf 
import os 
os.path.join('..')
import pandas as pd 
import matplotlib.pyplot as plt 

face_files = './data/lfw_attributes.txt'

# import numpy as np 
# faces = np.loadtxt(face_files,str,delimiter = '\t',unpack=True)
# print(faces.shape)
# print(faces[:,1])

import pandas as pd 
test = pd.read_table(face_files)
path = '.data/'
Asians = test[test['Asian']>1]
for person in Asians['person']:
	person = str(person).replace(' ','_')
	person_path = os.path.join(path,person)
	print(person_path)