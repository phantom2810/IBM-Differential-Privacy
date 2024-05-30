
#-------------------------------------------------------------#
# #inputs:
# # X_col_list=[]
# # Y_col_list=[]
# # tt_split=[] 
# # model_params=[eps,data_norm,clusters,bit] 
# #we can ask for a flag bit to indicate which model out of the 3 //then switch case

#OUR APPROACH
#get x_train,y_train,x_test,y_test
# get epsilon = 1 and data_norm = None.  parameters
# fit and compile the two models (dp and non dp)
# We can also visualise the tradeoff between accuracy and various epsilons using matplotlib. X

#OUTPUTS
#for classification op is accuracy
#for regression op is r2 error
#for clustering op is silihoutte index (labels – Index of the cluster the input sample belongs to. (any other evaluation metric))
#----------------------------------------------------------------------------#

import pandas as pd 
import diffprivlib.models as dp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from diffprivlib.models import LogisticRegression as dp_LogisticRegression, LinearRegression as dp_LinearRegression, KMeans as dp_KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def model_check(data_dict, csvfile):
    #Take csv from backend
    df= pd.read_csv(f"files/{csvfile}",header=0) #0 means top most row
    # print(df.shape) #(48842, 15)


    X_col_list = data_dict['colinp']
    Y_col_list=data_dict['colop']
    algo_choice = data_dict['mlalgo']
    tt_split=data_dict['traintest']
    test_size = tt_split[1]/100
    model_params=data_dict['mlpara']
    epsil_val=model_params[0]
    datanorm_val=model_params[1] #(The max l2 norm of any row of the data.)
    cluster_no=int(model_params[2])


    def preprocess(df):
        if(df.isnull().values.any()):
            df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    # df.replace(to_replace = np.nan, value =-99999) an alternative for later stages
        for col_name in df.columns:
            if(df[col_name].dtype == 'object'):
                df[col_name]= df[col_name].astype('category')
                df[col_name] = df[col_name].cat.codes
        return(df)

    def classification(df,X_train, X_test, y_train, y_test, epsil_val, datanorm_val):

        clf = LogisticRegression(solver="lbfgs")
        clf.fit(X_train, y_train.values.ravel())
        classification_baseline = clf.score(X_test, y_test)
        clf_accuracy = classification_baseline * 100

        # if datanorm_val==0: #ask for 0, if no l2 norm 
        #     datanorm_val='None'
        dp_clf = dp_LogisticRegression(epsilon=epsil_val,data_norm=datanorm_val)
        dp_clf.fit(X_train, y_train.values.ravel())
        dp_clf_accuracy = dp_clf.score(X_test, y_test) * 100

        return(clf_accuracy,dp_clf_accuracy)



    def regression(df,X_train, X_test, y_train, y_test, epsil_val):
        
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        regr_R2 = regr.score(X_test, y_test)
        
        dp_regr = dp_LinearRegression(epsilon=epsil_val)
        dp_regr.fit(X_train, y_train)
        dp_regr_R2 = dp_regr.score(X_test, y_test)
        return(regr_R2, dp_regr_R2)



    def clustering(df,X_train, X_test, y_train, y_test,cluster_no):
        # #might need data standardizing
        x = X_train.values
        kmeans = KMeans(n_clusters=cluster_no)
        y_kmeans = kmeans.fit_predict(x)
        # print(y_kmeans)
        silhouette_avg = silhouette_score(x, y_kmeans)

        dp_kmeans = dp_KMeans(n_clusters=cluster_no)
        dp_y_kmeans = dp_kmeans.fit_predict(x)
        # print(dp_y_kmeans)
        dp_silhouette_avg = silhouette_score(x, dp_y_kmeans)

        return(silhouette_avg, dp_silhouette_avg)


    prepped_df=preprocess(df)

    X = prepped_df[X_col_list].copy()
    Y = prepped_df[Y_col_list].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    # if algo_choice=
    # classification(prepped_df,X_train, X_test, y_train, y_test, epsil_val, datanorm_val)
    # regression(prepped_df,X_train, X_test, y_train, y_test, epsil_val)
    # clustering(prepped_df,X_train, X_test, y_train, y_test,cluster_no)


    switcher = {
        1: classification(prepped_df,X_train, X_test, y_train, y_test, epsil_val, datanorm_val),
        2: regression(prepped_df,X_train, X_test, y_train, y_test, epsil_val),
        3: clustering(prepped_df,X_train, X_test, y_train, y_test,cluster_no)
        }

    def switch(choice):
        return switcher.get(choice, -1)

    return switch(algo_choice)
