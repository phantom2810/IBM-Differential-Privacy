import warnings
import numpy as np
import pandas as pd
from diffprivlib.mechanisms import Laplace
from diffprivlib.mechanisms import Gaussian
from diffprivlib.mechanisms import Geometric
from diffprivlib.mechanisms import Binary
from diffprivlib.mechanisms import Exponential, ExponentialCategorical
import random

# {a1:[];a2:[]}

# #INPUT:
def noise_addd(data_dict, csvfile):
    col_list= data_dict['private'] #this is just for formality can cut
    binary_list=data_dict['binary']
    categ_list = data_dict['categorical']
    num_list=data_dict['numerical']

    print(num_list, csvfile)
    #we have to do error handling, what if no column is input empty list
    params_list=data_dict['epsilon']

    num_mech_choice=random.randint(1,3)
    local_epsilon = params_list[0]
    sensitivity = params_list[1] 
    delta_val = params_list[2]


    #***1st row has to be column names
    df= pd.read_csv(f"files/{csvfile}",header=0) #0 means top most row
    print(df.shape) #(48842, 15)
    rows= len(df. columns)

    def numerical_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val):

        if num_mech_choice==1:
    #LAPLACE
            a = Laplace(epsilon=local_epsilon, delta=delta_val, sensitivity=sensitivity)
            for j in num_list:
                for i in range(rows):
                    rando=a.randomise(df[j][i])
                    df.loc[i,j]=rando
                df[j] = df[j].astype(int)
        ## data like age cant be float so needed to be downcast to int by :

        elif num_mech_choice==2:
    #GAUSSIAN
            b = Gaussian(epsilon=local_epsilon, delta=delta_val, sensitivity=sensitivity)
            for j in num_list:
                for i in range(rows):
                    rando=b.randomise(df[j][i])
                    df.loc[i,j]=rando
                df[j] = df[j].astype(int)

        elif num_mech_choice==3:
    #GEOMETRIC
            c = Geometric(epsilon=local_epsilon, sensitivity=int(sensitivity)) #NO DELTA, SENSITIVITY SHOULD BE INT
            for j in num_list:
                for i in range(rows):
                    rando = c.randomise(df[j][i])
                    df.loc[i,j]=rando
                df[j] = df[j].astype(int)

    def binary_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val):
        #BINARY
        for j in binary_list:
            vals=df[j].unique() #np array
            b=Binary(epsilon=local_epsilon,value0=vals[0],value1=vals[1]) #new object for every column
            for i in range(rows):
                rando=b.randomise(df[j][i])
                df.loc[i,j]=rando

    def categorical_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val):
        #EXPONENTIAL
        import itertools
        util_list=[]

        def totuple(pair):
            a = pair
            x = random.random() #random probability
            new = (*a, x)
            util_list.append(new)

        for j in categ_list:
            l2= df[j].unique().tolist()
            for pair in itertools.product(l2, repeat=2):
                tuple=(pair)
                totuple(tuple)
            ec =  ExponentialCategorical(epsilon=local_epsilon,utility_list=util_list)
            for i in range(rows):
                rando=ec.randomise(df[j][i])
                df.loc[i,j]=rando


    numerical_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val)
    binary_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val)
    categorical_mech(df, num_mech_choice, local_epsilon, sensitivity, delta_val)


    # SAVING DP CSV
    df.to_csv(f"files/private_{csvfile}", index=False)