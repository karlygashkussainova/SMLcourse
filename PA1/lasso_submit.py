#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 6 23:06:30 2022

@author: karlykussainova
"""



import pandas as pd
import numpy as np
import sys
import random
import matplotlib
import matplotlib.pyplot as plt

from data_helper import generate_q4_data, load_simulate_data

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# load data
dpa = pd.read_csv('./data/house-votes-84.complete.csv')
dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
for i in range(16):
	index = 'A'+ str(i+1)
	dpa[index] = dpa[index].map({'y': 1, 'n': 0})
#dpa.info()

pay = dpa.Class
paX = dpa.drop('Class', axis = 1)



'''
  10-cv with house-votes-84.complete.csv using LASSO
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true 
  NOTE you do *not* need to modify this function
  '''
def lasso_evaluate(train_subset=False, subset_size = 0):
	sample_size = pay.shape[0]
	tot_incorrect=0
	tot_test=0
	tot_train_incorrect=0
	tot_train=0
	step = int( sample_size/ 10 + 1)
	for holdout_round, i in enumerate(range(0, sample_size, step)):
		#print("CV round: %s." % (holdout_round + 1))
		if(i==0):
			X_train = paX.iloc[i+step:sample_size]
			y_train = pay.iloc[i+step:sample_size]
		else:
			X_train =paX.iloc[0:i]  
			X_train = X_train.append(paX.iloc[i+step:sample_size], ignore_index=True)
			y_train = pay.iloc[0:i]
			y_train = y_train.append(pay.iloc[i+step:sample_size], ignore_index=True)
		X_test = paX.iloc[i: i+step]
		y_test = pay.iloc[i: i+step]
		if(train_subset):
			X_train = X_train.iloc[0:subset_size]
			y_train = y_train.iloc[0:subset_size]
		#print(" Samples={} test = {}".format(y_train.shape[0],y_test.shape[0]))
		# train the classifiers
		lasso = Lasso(alpha = 0.001)
		lasso.fit(X_train, y_train)            
		lasso_predit = lasso.predict(X_test)           # Use this model to predict the test data
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_test.values.tolist()[index] != num):
				error+=1
		tot_incorrect += error
		tot_test += len(lasso_result)
		#print('Error rate {}'.format(1.0*error/len(lasso_result)))
		lasso_predit = lasso.predict(X_train)           # Use this model to get the training error
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_train.values.tolist()[index] != num):
				error+=1
		tot_train_incorrect+= error
		tot_train += len(lasso_result)
		#print('Train Error rate {}'.format(1.0*error/len(lasso_result)))		

	#print('10CV Error rate {}'.format(1.0*tot_incorrect/tot_test))
	#print('10CV train Error rate {}'.format(1.0*tot_train_incorrect/tot_train))

	return 1.0*tot_incorrect/tot_test, 1.0*tot_train_incorrect/tot_train

def lasso_evaluate_incomplete_entry():
	# get incomplete data
	dpc = pd.read_csv('./data/house-votes-84.incomplete.csv')
	for i in range(16):
		index = 'A'+ str(i+1)
		dpc[index] = dpc[index].map({'y': 1, 'n': 0})
		
	lasso = Lasso(alpha = 0.001)
	lasso.fit(paX, pay)
	lasso_predit = lasso.predict(dpc)
	print(lasso_predit)


def main():
	#For Q2
	print('____________________________')
	print('Question 2:')
	error_rate, unused = lasso_evaluate()
	print('10-fold cross validation error is {}'.format(error_rate))

	
	#For Q3
	print('____________________________')
	print('Question 3: LASSO (Small Data). Please, see question_3_Lasso.png for plots')
	train_error = np.zeros(10)
	test_error = np.zeros(10)
	example_number = np.zeros(10)

	for i in range(10):
		example_number[i] = (i+1)*10
		x, y =lasso_evaluate(train_subset=True, subset_size=i*10+10)
		train_error[i] = y
		test_error[i] = x

	plt.figure()
	plt.subplot(211)
	plt.plot(example_number, train_error)
	plt.ylabel('train error')
	plt.title('LASSO')

	plt.subplot(212)
	plt.plot(example_number, test_error)
	plt.ylabel('test error')
	plt.xlabel('sample size')
	plt.savefig('question_3_Lasso.png')
	plt.show()
	
	
	#Q4
	#TODO 
	print('____________________________')
	print('Question 4: Generating samples (see new_q4_data_Lasso_.csv files) and a plotting the fraction of nonpartisan bills(see Lasso_q4.png)')

	#random.seed(1234)
	#np.random.seed(1234)
	EPS = sys.float_info.epsilon
	nonpartisian = 12
	training_set_size = []
	fraction_ign_lst = []
    
	for i in range(400, 4001, 400):
		training_set_size.append(i)
		file_name = "new_q4_data_Lasso{}.csv".format(i)
		generate_q4_data(i, file_name)

		# load data
		dpa = pd.read_csv(file_name)
		dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
		for i in range(16):
			index = 'A'+ str(i+1)
			dpa[index] = dpa[index].map({'y': 1, 'n': 0})
		pay = dpa.Class
		paX = dpa.drop('Class', axis = 1)

        # Apply Lasso
		lasso = Lasso(alpha = 0.001)
		lasso.fit(paX, pay)

		coefs = lasso.coef_

		
		nonpartisian_ignored = 0
		for j in range(4, 16):
			if abs(coefs[j])<EPS:
				nonpartisian_ignored = nonpartisian_ignored + 1
				
		fraction_ignored = float(nonpartisian_ignored)/nonpartisian
		fraction_ign_lst.append(fraction_ignored)

	# Plot
	plt.figure()
	plt.plot(training_set_size, fraction_ign_lst)
	plt.title('Fraction of ignored nonpartisan bills (LASSO)')
	plt.xlabel('training set size')
	plt.ylabel('Fraction of ignored')
	plt.savefig('Lasso_q4.png')
	#plt.show()
	

	#Q5
	print('____________________________')
	print('Question 5:')
	print('LASSO  P(C=1|A_observed) ')
	lasso_evaluate_incomplete_entry()
	

if __name__ == "__main__":
    main()