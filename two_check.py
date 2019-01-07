import numpy as np
import pandas
import csv
import matplotlib.pyplot as plt
import keras
import random
import keras.utils
from keras import utils as np_utils
from imblearn.over_sampling import SMOTE, ADASYN
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.layers.normalization import BatchNormalization

class Credit_Risk(object):
    def __init__(self):
        self.seed = 7
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
	self.y_predict = None

    ## encode class values as integers
    #encoder = LabelEncoder()
    #encoder.fit(Y)
    #encoded_Y = encoder.transform(Y)
    ## convert integers to dummy variables (i.e. one hot encoded)
    #dummy_y = keras.utils.to_categorical(encoded_Y)
    

    def load_data(self):
	for node in range(5):
		for node2 in range(5):
			for batch in range(5):
				for epoc in range(5): 
					sum1 = np.zeros((6,))
					print("epoch")
					for loop in range(6):
		
						seed2 = 9
						np.random.seed(seed2)
						#training data

						dataframe = pandas.read_csv("train"+str(loop)+".csv", header=None)
						dataset = dataframe.values
						temp1 = dataset[:,:].astype(float)
						print(temp1.shape)
						np.random.shuffle(temp1)
						Y1 = temp1[:,14:15].astype(int)
						X = temp1[:,2:14].astype(float)
						X -= np.mean(X, axis = 0)
						X /=np.std(X,axis = 0)
						Y1 = Y1.reshape(-1)
						X , Y1 = SMOTE().fit_sample(X, Y1)
						Y = keras.utils.to_categorical(Y1)

						dataframe = pandas.read_csv("test"+str(loop)+".csv", header=None)
						dataset = dataframe.values
						full = dataset[:,:].astype(float)
						company = dataset[:,0:1].astype(int)
						deafult = dataset[:,15:16].astype(int)
						time = dataset[:,1:2].astype(int)
						Y2 = dataset[:,14:15].astype(int)
						X2 = dataset[:,2:14].astype(float)
						X2 -= np.mean(X2, axis = 0)
						X2 /=np.std(X2,axis = 0)	
						Y2 = Y2.reshape(-1)
						y_test = keras.utils.to_categorical(Y2)
						self.x_train = X
						self.y_train = Y
						self.y_test = y_test
						self.x_test = X2

					   
						np.random.seed(self.seed)
						# create model
						model = Sequential()
						model.add(Dense(5*(node+1), input_dim=12, activation='relu'))
						model.add(BatchNormalization())
						model.add(Dropout(0.5))
						model.add(Dense(5*(node2+1), activation='relu'))
						model.add(BatchNormalization())
						model.add(Dropout(0.5))
						model.add(Dense(4, activation ='softmax'))
						# Compile model
						b =keras.optimizers.Adam(lr= .01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

						model.compile(loss= 'categorical_crossentropy' , optimizer='adam', metrics=['accuracy'])
						self.model = model
						self.model.fit(self.x_train, self.y_train, epochs= 100*(epoc+1), batch_size = 300*(batch+1))
	
						score = self.model.evaluate(self.x_test, self.y_test)
						print(score)
					    	self.y_predict = self.model.predict(self.x_test)
						#print (self.y_predict.shape)
						final = []
						coun = 1
						for loop1 in range(12):
							if(loop == 5):
								if(loop1 >=6):
									break
							var = []

							for i in range(company.shape[0]):			
								if( time[i] % 100 == loop1+1):
									var.append(full[i])
									#print(company[i])
									#coun+=1
							var = np.array(var)
							def0 =[]
							coun = 1
							print(var.shape)
							for i in range(var.shape[0]):
								if(var[i][15] == 1):
									coun+=1
									def0.append(var[i][0])

							def0 = np.array(def0)
							#print(def0.shape)
							hmm = var[:, 2:14]
							hmm -= np.mean(hmm, axis = 0)
							hmm /=np.std(hmm,axis = 0)
							res1 =self.model.predict(hmm)


							def1 = np.zeros((def0.shape[0], ))
							def2 = np.zeros((def0.shape[0], ))
							def3 = np.zeros((def0.shape[0], ))
							def4 = np.zeros((def0.shape[0], ))
							def4 = np.zeros((def0.shape[0], ))
							def5 = np.zeros((def0.shape[0], ))
							temp = np.zeros((6,))
							b = []
							for i in range(var.shape[0]):
								b.append(res1[i][1])
							b = np.array(b)
							b = np.sort(b)
							b = np.flip(b, 0)
							#np.savetxt("exit1.csv", b, delimiter=",")
	
							for j in range(def0.shape[0]):
								for i in range(var.shape[0]):
									if( var[i][0] == def0[j]):
										if(res1[i][1] >= b[(b.shape[0]/10)]):
											def1[j] = 1 
											#print("lol")
										if(res1[i][1] >= b[(b.shape[0]*2)/10]):
											def2[j] = 1 
										if(res1[i][1] >= b[(b.shape[0]*3)/10]):
											def3[j] = 1 
										if(res1[i][1] >= b[(b.shape[0]*4)/10]):
											def4[j] = 1 
										if(res1[i][1] >= b[(b.shape[0]*5)/10]):
											def5[j] = 1 
					
							print("number of default")
							print(def0.shape[0])
							for i in range(def0.shape[0]):
								temp[0] += def1[i]
								temp[1] += def2[i]
								temp[2] += def3[i]
								temp[3] += def4[i]
								temp[4] += def5[i]
	
							temp[5] = def0.shape[0]
							for i in range(6):
								print(temp[i])
								sum1[i]+=temp[i]
							np.savetxt("exit1.csv", c, delimiter=",")
							
						for q in range(6):
							print(sum1[q])
					np.savetxt("two"+str(node)+"_"+str(node2)+"_"+str(epoc)+"_"+str(batch)+".csv", sum1, delimiter=",")
if __name__ == '__main__':
    credit = Credit_Risk()
    credit.load_data()



