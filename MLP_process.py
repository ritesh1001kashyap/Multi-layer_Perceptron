'''
Created on Apr 24, 2019

@author: ritesh
'''

'''
we are importing hiddenNode class. so that we can inherit all the properties of 
perceptron object and use it in calculating optimized weights

importing xlrd library to read input data from excel 
'''
from hiddenNode import hiddenNode
import xlrd


loc = ("E:\\ATF\\DATA.xlsx") #loc refers to location of excel file which stores stock price ddata

wb = xlrd.open_workbook(loc) #wb is an object which refers to above excel 
sheet = wb.sheet_by_index(2)# sheet object refers to sheet number 2 of the above workbook


def topology():
    
    '''
    topology() function is user interface function, which is used to accept values from user.
    The function will ask following values from user:
    1- number of input variable
    2- number of nodes in input layer
    3- number of hidden layer
    4- number of nodes in each hidden layer
    
    all the data provided by user will be stored in a list named as topology _list, which will 
    act as a base for topology creation
    '''
    
    topology_list=[0] # initializing topology list with 1st value as zero
    print("enter number of input variables")
    n=input() # n will store number of input variables in topology, provided by user
    n=int(n)
    topology_list[0]=n  # the first value of topology_list will store number of input variable in topology
    
    print("enter number of input nodes")
    input_nodes=input() #accepting value from user
    '''
    input_nodes will store number of input nodes in the topology, given by the user.
    Second value of topology_list will store number of input nodes in the topology
    '''  
    input_nodes=int(input_nodes) #converting value to integer
    topology_list.append(input_nodes) # appending input_nodes to topology_list
    
    print("enter number of hidden layer")
    '''
    hidden_layer will accept number of hidden layers to be incorporated in the topology
    '''
    hidden_layer=input()  #accepting value from user
    hidden_layer=int(hidden_layer) # converting to integer 
    
    for i in range(0,hidden_layer): # i is the iteration variable
        '''
        this loop is for accepting number of nodes for each hidden layer
        each value will be appended in topology_list
        '''
        print("enter number of nodes in "+str(i+1)+"th hidden layer")
        nodes=input() #accepting value from user
        nodes=int(nodes) # converting to integer
        topology_list.append(nodes)
    '''
    this function will return the topology_list
    '''    
    return topology_list

def nodes_object_creation(topology_list):
    '''
    topology_object is a 2d array that will store objects of nodes of all layers of topology
    layer_object is a 1d array that will store object of nodes for each layer
    '''
    topology_object=[0]*(len(topology_list)-1) 
    
    '''
    topology_object is a 2D array that will store hiddenNode Object for each nodes,
    based on topology data provided by user in topology_list
    '''
    
    for i in range(1,len(topology_list)): # i is iteration variable
        layer_objects=[0]*topology_list[i]
        '''
        layer_objects is a 1D array, that will store objects corresponding to each layer
        after each iteration, layer_objects will get appended in topology_object
        '''
        for j in range(0,topology_list[i]):  # j is the iteration loop
            '''
            this loop will for creating objects and inheriting all the properties of node objects
            '''
            n=topology_list[i-1]+1
            input_list=[1]*n  #initializing input_list with initial value 1 for each object
            weight_list=[0.2]*n #initializing weight_list with initial value 0.2 for each object
            derivative_list=[0]*n  #initializing derivative_list with initial value 0 for each object
            obj=hiddenNode(n,input_list,weight_list,derivative_list) #creating object from hiddenNode
            
            layer_objects[j]=obj # storing obj in layer_objects list
            
        topology_object[i-1]=layer_objects   # appending layer_objects in topology_list
    '''
    creation of output node:
    In our topology there will be only one output node.
    we are creating output node object separately.
    output node will be the last object in topology_object
    '''
    out=topology_list[len(topology_list)-1]+1  #adding 1 for bias
    input_list=[1]*out   #initializing input_list with initial value 1 for each object
    weight_list=[0.2]*out   #initializing weight_list with initial value 0.2 for each object
    derivative_list=[0]*out  #initializing derivative_list with initial value 0 for each object
    objout=hiddenNode(out,input_list,weight_list,derivative_list) #creating object from hiddenNode
    layer_objects=[objout]
    
    topology_object.append(layer_objects) #appending output node to topology_object
    
    '''
    this function will return topology_object as output
    '''
    return topology_object




def input_updation(topology_object, topology_list):
    '''
    this function will update inputs of nodes based on topology of MLP
    The idea behind input updation is: output from nth layer is input for (n+1)th layer.
    So we have to calculate output of each node, and then update input for each (n+1)th node
    '''
    for i in range(1, len(topology_object)): # i refers to layer number of the topology
        for j in range(0,len(topology_object[i])): #j refer to perceptron number in ith layer
            
            for k in range(0,len(topology_object[i-1])): # k refers to kth input variable of jth perceptron
                topology_object[i][j].input_list[k]=topology_object[i-1][k].node_output()
                '''
                in the loop we are updating input_list of each node,
                based on output values coming from previous layer
                '''


def derivative_updation(topology_object, topology_list,actual_value):
    '''
    this function will calculate derivative of error with respect to each weights
    '''
    path_derivative=1
    derivative=0
    '''
    path_derivative will store derivative between two nodes along one path
    derivative is sum of all path_derivative.
    basically derivative consist of sum of all possible path_derivatives between two nodes 
    '''
    for i in range(0,len(topology_object)): # i refers to layer number of the topology
        for j in range(0, len(topology_object[i])):  # j refers to jth perceptron in ith layer
            for k in range(0,len(topology_object[i][j].weight_list)):  # k refers kth weight in jth perceptron
                
                for l in range(0,len(topology_object)-1): # l refers to 'source' layer number
                    for m in range(0, len(topology_object[l])): # m refers to 'destination' layer number
                        for n in range(0, len(topology_object[l+1])): # n refers to all the possible path available between source and destination node
                            path_derivative=path_derivative*topology_object[l][m].node_output()*(1-topology_object[l][m].node_output())*topology_object[l+1][n].weight_list[n]
                            
                        derivative=derivative+path_derivative
                derivative=derivative*(-2)*(actual_value- topology_object[len(topology_object)-1][0].node_output())*topology_object[i][j].input_list[k]
                topology_object[i][j].derivative_list[k]=derivative
                '''
                after calculating path_derivative and derivative, derivative _list of each perceptron object
                will be updated.
                '''

def back_propagation(topology_object,topology_list,actual_value, learning_rate):
    
    '''
    this function is for back propagation and updating weights based on derivative and learning rate
    '''
    
    m=len(topology_object)-1  # m refers to layer number of output node
    n=len(topology_object[m])-1 # n refers to perceptron number in output layer
    
    error=(actual_value-topology_object[m][n].node_output())*(actual_value-topology_object[m][n].node_output())
    '''
    error will store square of difference of actual and predicted value.
    predicted value is output of last node.
    '''
    
    while(error>0.003):
        '''
        this while loop is for back propagation, with error threshold of 0.003.
        whenever the error is greater than 0.003, the loop will execute and update weights and output.
        At the end of the loop, we will again calculate the error based on updated weights and output
        '''
        for i in range(0,len(topology_object)): # i refers to ith layer of topology
            for j in range(0,len(topology_object[i])):  #j refers to jth perceptron in ith layer
                for k in range(0,len(topology_object[i][j].weight_list)):  # k refers to kth weight list of jth perceptron
                    
                    topology_object[i][j].weight_list[k]=topology_object[i][j].weight_list[k] - topology_object[i][j].derivative_list[k]*learning_rate
        '''
        after updating weights of each nodes, ww will call input_updation() function, in order to
        update input of each node, based on output received from previous layer
        '''            
        input_updation(topology_object, topology_list)
        
        '''
        calculating updated error at the end of the loop
        '''
        error=(actual_value-topology_object[m][n].node_output())*(actual_value-topology_object[m][n].node_output())
        print(error)
def main():
    '''
    the main function is like a central processing unit of MLP.
    It will decide the order of functions to be called.
    In other words, this function decides the flowchart of MLP
    '''
    topology_list=topology()
    object1=nodes_object_creation(topology_list)
    '''
    calling topology() function in order to get topology_list based on values entered by user
    '''
    
    '''
    then we will call input_updation() function and derivative_updation() function in order to
    update input_list and derivative_list of each node, based on initial values of input and weights.
    '''
    #print(object1[1][0].node_output())
    input_updation(object1, topology_list)
    #print(object1[1][0].node_output())
    #print(object1[3][0].derivative_list)
    derivative_updation(object1, topology_list,3)
    #print(object1[3][0].derivative_list)
    
    
    print("enter learning rate")
    learning_rate=input() #accepting learning rate from user
    learning_rate=float(learning_rate)  # converting to float
    print("before",object1[0][0].input_list)
    number_of_variable=topology_list[0] # number_of_variable will store number of variable to be used in topology
    input_list=[1]*(number_of_variable+1)
    
    
    
    for i in range(0, 500):
        '''
        this loop is for reading stock price data from excel and using it as input variable
        '''
        for j in range(0,number_of_variable):
            input_list[j]=sheet.cell_value((i)+j+1,7) # preparing input_list for input node
        actual_value=sheet.cell_value((i)+number_of_variable,7) # extracting actual value form excel
        
        
        '''
        updating input_list of all input nodes
        '''
        print("after",input_list)
        for i in range(0,len(object1[0])):
            object1[0][i].input_list=input_list
         
         
        '''
        after extracting data from excel, we will again call input_updation(), derivative_updation() and 
        back_propagation() function
        '''    
        input_updation(object1, topology_list)
        
        derivative_updation(object1, topology_list, actual_value)
        
        back_propagation(object1, topology_list, actual_value, learning_rate)
        
        print(object1[3][0].node_output())
main()

'''
In the end calling main() function.

MAY THE FORCE BE WITH US
'''
