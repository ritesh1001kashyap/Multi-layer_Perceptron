'''
Created on Apr 24, 2019

@author: ritesh
'''
import math
class hiddenNode:
    '''
    
    This class will create object for all perceptrons present in the topology.
    each object of perceptron will have the following properties:
    1- number of inputs entering into perceptron
    2- list of input variable
    3- list of weights corresponding to each input variable
    4- derivative of error with respective to each weights
    '''


    def __init__(self, number_of_inputs, input_list, weight_list, derivative_list):
        '''
        this constructor will initialize object of class with all the properties.
        number_of_inputs: number of input variable entering into perceptron
        input_list: list of all input variables and one bias term
        weight_list: it is a list which contains weights for each input variable
        derivative_list: it is a list which contains derivative of error with respect to each weights
        '''
        self.input_list=input_list
        self.weight_list=weight_list
        self.derivative_list=derivative_list
        self.number_of_inputs=number_of_inputs
        
    
        
    def node_output(self):
        '''
        this function will calculate output of the node, based on 
        input variable and weights received
        initial value of output is zero
        '''
        output=0
        for i in range(0,len(self.input_list)): #here i is itiration variable
            output=self.input_list[i]*self.weight_list[i] + output
            '''
            the loop will calculate sum of product of weights and input variable
            this output will go as input in sigmoid function
            '''
        if(output<0):
        # if condition to avoid math overflow error
            output=1-1/(1+math.exp(output))
        else:
            output=1/(1+math.exp(-1*output))
        '''
        after calculating output of sigmoid function, 
        the function will return the value of sigmoid function
        '''  
        return output
            
