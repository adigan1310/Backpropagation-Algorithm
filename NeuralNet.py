import pandas as pd
import numpy as np
import csv
import math
import sys
import random

etolerance = 0.0
toterror = 0.0
tr = 0.5

# Class for layers
class Layers(object):
    def __init__(self):
        self.id = 0
        self.isinput = True
        self.ishidden = False
        self.isoutput = False
        self.nodes = 0
        self.neurons = list()

# Class for neurons
class Neurons(object):
    def __init__(self):
        self.layerId = 0
        self.id = 0
        self.bias = 0.0
        self.value = 0.0
        self.delta = 0.0
        self.ipweights = {}

# Function to create input neurons
def getInputNeurons(index, reader):
    iden = 0
    neurons = list()
    for i in range(0,index):
        iden = iden + 1
        n1 = Neurons()
        n1.layerID = 1
        n1.id = iden
        neurons.append(n1)
    return neurons

# Function to get random value for weights
def randomval():
    return random.uniform(-0.5,0.5)

# Function to generate random weights to a neuron
def generateweights(layerid):
    weights = {}
    for i in range (1,Globalmap[layerid].nodes+1):
        weights[i] = randomval()
    return weights

# Function to print constructed network
def printNet():
    keys = Globalmap
    for i in range(1,len(keys)):
        count = 0
        if i == 1:
            print("Layer " + str(i-1) + "(Input Layer): ")
        elif i == len(keys)-1:
            print("Layer " + str(i - 1) + "(Last hidden layer): ")
        else:
            print("Layer " + str(i - 1) + "(" + str(i - 1) + " hidden layer): ")
        lst = keys[i].neurons
        for n in lst:
            count = count + 1
            print("Neuron " + str(count) + " weights:", end="")
            for neus in keys[i+1].neurons:
                print(str(neus.ipweights[count]) + " , ",end="")
            print("\b\b\b")
        print("")

# Function to perform back propagation algorithm
def backprop(trainingdata,iterations):
    global toterror
    global otcount
    stop = False
    count = iterations
    while stop is False and count > 0:
        sum_error = 0
        for row in trainingdata:
            initinput(row,Globalmap[1])
            outputs = fpass(row)
            expected = [0 for i in range(otcount)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            bpass(row, expected)
        if 0.5*sum_error == 0:
            stop = True
        count = count - 1
        toterror = 0.5*sum_error

# Function to initialize the inputs
def initinput(instance, node):
    inputs = node.neurons
    for i in range(0,len(instance)-1):
        for row in inputs:
            if int(row.id) == i+1:
                row.value = float(instance[i])

# Function to perform forward pass
def fpass(instance):
    outvalue = list()
    for i in range(1,len(Globalmap)+1):
        if Globalmap[i].isinput == False:
            for ns in Globalmap[i].neurons:
                val = findoutput(ns,Globalmap[Globalmap[i].id - 1])
                ns.value = val
                if Globalmap[i].isoutput == True:
                    outvalue.append(val)
    return outvalue

# Function to perform backward pass
def bpass(instance, expected):
    global tr
    iter = len(Globalmap)
    for i in range(iter , 0 , -1):
        Lyr = Globalmap[i]
        for ns in Lyr.neurons:
            ns.delta = finddelta(Lyr, expected,ns)
            if Lyr.isoutput is True or Lyr.ishidden is True:
                ns.bias = ns.bias + (ns.delta*tr)

# Function to calculate delta value
def finddelta(Lyr, input, nes):
    global tr
    x = nes.value
    i = int(nes.id)
    if Lyr.isoutput is True:
        d = float(x * (1 - x) * (input[i-1] - x))
    else:
        sm = 0.0
        l = Globalmap[Lyr.id+1]
        for ns in l.neurons:
            sm = sm + (ns.delta * ns.ipweights[i])
            currweight = ns.ipweights[i]
            updatedweight = float(currweight + float(ns.delta*x*tr))
            ns.ipweights[i] = updatedweight
        d = float(x * (1 - x) * sm)
    return d

# Function to calculate output value of a neuron
def findoutput(ns, prevlayer):
    sum = 0.0
    neuronslist = prevlayer.neurons
    for row in neuronslist:
        i = row.id
        sum = sum + (row.value * ns.ipweights[i])
    sum = sum + ns.bias
    return sigmoid(sum)

# Function to calculate sigmoid value
def sigmoid(value):
    if value < 0:
        val = 1 - 1/(1+math.exp(value))
    else:
        val = 1 / (1 + math.exp(-value))
    return val

# Function to test unseen dataset
def test(testingdata):
    global otcount
    sum_error = 0
    for row in testingdata:
        initinput(row,Globalmap[1])
        outputs = fpass(row)
        expected = [0 for i in range(otcount)]
        expected[int(row[-1])] = 1
        sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
    sum_error = sum_error*0.5
    print("Total Test Error: " + str(sum_error))

# Function to construct input layer
def init_input(attributes,training_data):
    layer = Layers()
    layer.id = 1
    layer.isinput = True
    layer.isoutput = False
    layer.ishidden = False
    neu = getInputNeurons(attributes - 1, training_data)
    layer.neurons = neu
    layer.nodes = len(neu)
    return layer

# Function to create hidden layer neurons
def createhiddenlayer(layerid,neuronid):
    neurons = Neurons()
    neurons.ipweights = generateweights(layerid-1)
    neurons.layerID = layerid
    neurons.id = neuronid
    neurons.bias = randomval()
    return neurons

# Function to construct output layer
def init_output(Layerid):
    global otcount
    output = Layers()
    output.isinput = False
    output.ishidden = False
    output.isoutput = True
    output.id = Layerid
    neuronID = 0
    out = list()
    for i in range(0, otcount):
        neuronID = neuronID + 1
        outputnode = Neurons()
        outputnode.id = neuronID
        outputnode.layerID = Layerid
        outputnode.bias = randomval()
        outputnode.ipweights = generateweights(Layerid - 1)
        out.append(outputnode)
    output.neurons = out
    output.nodes = len(out)
    return output

training_data = list()
testing_data = list()
count = 4
# To read the preprocessed data file and the requirements for constructing the network
List = input().split()
for i in range(0,int(List[3])):
    List[count] = int(List[count])
    count = count + 1
reader = pd.read_csv(List[0],header=None,skiprows=1)
attributes = len(reader.columns)
otcount = reader[attributes-1].nunique()
data = reader.values.tolist()
random.shuffle(data)
training_index = int((int(List[1])*(len(data))/100))
training_data = data[:training_index]
testing_data = data[training_index:]
iterations = int(List[2])
hiddenlayers = int(List[3])
Globalmap = {}
Layerid = 1;
# Construct Input Layer
Globalmap[Layerid] = init_input(attributes,training_data)
# Construct Hidden Layers
count = 4
for i in range(0, hiddenlayers):
    neuronID = 0
    Layerid = Layerid + 1
    hidden = Layers()
    hidden.id = Layerid
    hidden.isinput = False
    hidden.ishidden = True
    hidden.isoutput = False
    noofneurons = List[i + count]
    neuronslist = list()
    for i in range(0, noofneurons):
        neuronID = neuronID + 1
        neuronslist.append(createhiddenlayer(Layerid, neuronID))
    hidden.neurons = neuronslist
    hidden.nodes = len(neuronslist)
    Globalmap[Layerid] = hidden
Layerid = Layerid + 1
# Construct Output Layer
Globalmap[Layerid] = init_output(Layerid)
# Print the constructed network
printNet()
# Perform back propagation algorithm
backprop(training_data,iterations)
print("Total Training Error: " +str(toterror))
# Test the unseen data with the network
test(testing_data)
















