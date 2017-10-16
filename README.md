"# Backpropagation-Algorithm" 

BACKPROPAGATION ALGORITHM IMPLEMENTATION

Language used : Python

Requirements: Any data set downloaded as txt/data/csv file

Steps:
1) Execute PreProcess.py for scaling the data
2) Provide the input file path and output file path 
	Example: F:/###.txt F:/###.csv
3) The data will be processed and stored in a csv file.
4) Execute NeuralNet.py 
5) Provide the parameters <Input File path name> <% of data to be split for training> <# of iterations> <# of hidden layers> <# of neurons in each hidden layer>
	Example: F:/###.csv 80 200 2 4 2
6) The code will construct the neural network and execute the back propagation algorithm for updating the weights. 
7) It will use the testing data to test it with the neural network and prints the training and testing error.

Author: Adithya Ganapathy
