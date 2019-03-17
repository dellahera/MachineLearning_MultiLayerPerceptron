import pandas as pd
import numpy as np
from random import seed
from functools import reduce
from random import random
import matplotlib.pyplot as plt 
import math as m
df = pd.read_csv('iris.csv')
idx = ['x1','x2','x3','x4','name']
df= pd.read_csv('iris.csv',names=idx)
df.loc[df['name']== 'setosa', 'code']=0
df.loc[df['name']== 'versicolor', 'code']=1
df.loc[df['name']== 'virginica', 'code']=2
df = df.drop(['name'], axis = 1)

valid = df[:10].append(df[51:60].append(df[101:110], ignore_index= True),ignore_index=True).head(30).values.tolist()
training = df[10:50].append(df[60:100].append(df[110:150], ignore_index= True),ignore_index=True).head(120).values.tolist()
seed(0.9)
n_inputs = 4
n_outputs = 3
n_hidden = 6 
n_epoch = 300
learning_rate = 0.1
skema=[]
hidden_layer = [{'theta':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
skema.append(hidden_layer)
output_layer = [{'theta':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
skema.append(output_layer)
plot_error=[]
plot_accuracy=[]
plot_errorv=[]
plot_accuracyv=[]

#function
def total(inp, weight):
    totalWeight= weight[-1]
    for i in range(len(weight)-1):
         totalWeight += weight[i]*inp[i]
    return totalWeight

def activate(totalWeight):
    return 1.0 / (1.0 + m.exp(-totalWeight))

def upd_weight(skema, row, learning_rate):
    for i in range(len(skema)):
        inputs = row[:-1]		
        if i != 0:
            inputs = [node['output'] for node in skema[i - 1]]
        for node in skema[i]:
            #Bagian update theta dan bias
            for j in range(len(inputs)):
                node['theta'][j] += learning_rate * node['d'] * inputs[j]
            node['theta'][-1] += learning_rate * node['d']
			
def forward(skema, row):
	current = row
	for layer in skema:
		after = []
		for node in layer:
			node['output'] = activate(total(node['theta'],current))
			after.append(node['output'])
		current = after
	return current
	
def backward(skema, target):
	#Loop mundur dari layer output ke layer hidden
	for i in range(len(skema)-1,-1,-1):
		layer = skema[i]
		errors = list()
		if i != len(skema)-1:
			for j in range(len(layer)):
				error = 0.0
				for node in skema[i + 1]:
					error += (node['theta'][j] * node['d'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				node = layer[j]
				errors.append(target[j] - node['output'])
		for j in range(len(layer)):
			node = layer[j]
			node['d'] = errors[j] * (node['output'] * (1.0 - node['output']))
	
def process_train(skema, training, learning_rate, n_outputs):
	returned = []
	error_total = 0
	correct_total = 0
	for row in training:
		outputs = forward(skema, row)
		target = [0 for i in range(n_outputs)]
		target[int(row[-1])] = 1
		for i in range(len(target)):
			error_total += m.pow((target[i]-outputs[i]),2)*0.5
		for i in range(len(target)):
			if outputs[i] > 0.5:
				outputs[i] = 1
			else:
				outputs[i] = 0
		if (outputs == target):
			correct_total+=1
		backward(skema, target)
		upd_weight(skema, row, learning_rate)
	returned.append(error_total/len(training))
	returned.append(correct_total/len(training))
	return returned
	
def process_validate(skema, training, n_outputs):
	returned = []
	error_total = 0
	correct_total = 0
	for row in training:
		outputs = forward(skema, row)
		target = [0 for i in range(n_outputs)]
		target[int(row[-1])] = 1
		for i in range(len(target)):
			error_total += m.pow((target[i]-outputs[i]),2)*0.5
		for i in range(len(target)):
			if outputs[i] > 0.5:
				outputs[i] = 1
			else:
				outputs[i] = 0
		if (outputs == target):
			correct_total+=1
		backward(skema, target)
	returned.append(error_total/len(training))
	returned.append(correct_total/len(training))
	return returned
	
def PrintGrafError():
    plt.plot(plot_error,label='Training')
    plt.plot(plot_errorv,label='Validation')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
	
def PrintGrafAcc():
    plt.plot(plot_accuracy,label='Training')
    plt.plot(plot_accuracyv,label='Validation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

	#main program
for epoch in range(n_epoch):
	tmp_error = 0.0        
	tmp_accuracy = 0.0
	tmp_error_v = 0.0      
	tmp_accuracy_v = 0.0
	print(epoch)
	t = process_train(skema, training, learning_rate, n_outputs)
	tmp_error+=t[0]
	tmp_accuracy+=t[1]
	v = process_validate(skema,valid,n_outputs)
	tmp_error_v = v[0]
	tmp_accuracy_v = v[1]
	plot_error.append(tmp_error)
	plot_accuracy.append(tmp_accuracy)
	plot_errorv.append(tmp_error_v)
	plot_accuracyv.append(tmp_accuracy_v)
	#show graph
PrintGrafError()
PrintGrafAcc()