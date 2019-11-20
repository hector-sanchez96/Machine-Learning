
from scipy.linalg import solve
from numpy.linalg import cholesky
from scipy.optimize import minimize
from numpy.linalg import lstsq
from  numpy.random import randn as rd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import Loading_and_cleansing_01 as ld 
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV





# =============================================================================
# 
# 
#                          K-NEAREST NEIGHBOR
# 
# =============================================================================


def euclidean(training_samples, query):
    
    distance= np.zeros(len(training_samples.iloc[:,0]))
    
    
    for each in range(len(training_samples.columns)):
        try:
            distance += pow(training_samples.iloc[:,each] - query.iloc[0,each], 2)   
        except:
                print("Something has gone wrong in the calculation of the eucledian distance")
            
    distance= np.sqrt(distance)
    
    return distance





def fit_knn_forecasting(x_train, y_train, x_test, y_test, window_s=100, horizon=50):
    
    best=0
    train_MSE=0
    test_MSE=0
    y_hat=0
    cv_results=0
    
    if len(x_train.iloc[:,0]) < 500:
        print('You need at least 500 observations in your training set to perform forward chaining cross validation')
        
    else:
        #Automatically calculate prediction based on best possible K:
        #YOU NEED AT LEAST 500 OBSERVATIONS IN YOU TRAINING SET!
        
        #Manual implementation of forward chaining cross-validation: 
        #Our default number of folds is 10
        print('Phase 1: Trying to find the best K for your data...')
                
        validation= pd.DataFrame(np.zeros(shape=(9,5)), index=np.arange(1,10), columns=['fold', 'forecast', 'best_k', 'MSE', 'num_of_obs' ])
        validation['fold']= np.arange(1,10)#change row and column indexes
        
        #if len(x_train.iloc[:,0]) > 100:
        #    x_train = pd.DataFrame(x_train.iloc[-100:,:], copy=True)
        #    y_train = pd.DataFrame(y_train.iloc[-100:,:], copy=True)
        
        all_errors= pd.DataFrame(index=np.arange(0, 9))
        all_errors['k']=np.array([2,3,5,7,9,15,25,50,75])
        all_errors['avg']=np.array([0,0,0,0,0,0,0,0,0])
        
        possible_k= np.array([2,3,5,7,9,15,25,50,75])
        
        print('Forward chaining cross validation has started')
        
        for fold in range(10,100,10):
    
            upper_bound = int((fold* len(x_train))/100)#max number of observations for each fold
            
            #fold n:
            fold_k_x= pd.DataFrame(x_train.iloc[0:upper_bound,:], copy=True)#at each iteration the the fold is increased
            fold_k_y= pd.DataFrame(y_train.iloc[0:upper_bound,:], copy=True)
            #left-out fold: 
            next_query_x= pd.DataFrame(x_train.iloc[[upper_bound],:])
            next_query_y= pd.DataFrame(y_train.iloc[[upper_bound],:])
            
            print('Evaluating new fold')
            
            distances= pd.DataFrame()
            distances['distance_to_query']= euclidean(fold_k_x, next_query_x)
            distances= pd.concat( [fold_k_x, distances, fold_k_y], axis=1)
            data_with_distances= distances.sort_values('distance_to_query')# distances in descending order
            
        
            possible_k=np.array([2,3,5,7,9,15,25,50,75])
            
            top_k= pd.DataFrame(index=np.arange(0, 9))
            top_k['k']= possible_k#first column is K (NN)
            top_k['forecast']=np.ones(9)
            top_k['error']=np.ones(9)
            
            
            for i in range(len(possible_k)):
                
                k_neighbors= pd.DataFrame(data_with_distances.iloc[0:top_k.iloc[i,0], :], copy=True)
                forecast= sum(k_neighbors.iloc[0:top_k.iloc[i,0], -1]) / (i+1)
                RMSE = np.sqrt(mean_squared_error(next_query_y.values, [forecast]))
                
                top_k.iloc[i, 1]=forecast
                top_k.iloc[i, 2]=RMSE
                
            all_errors= all_errors.merge(pd.DataFrame(top_k['error']), left_index=True, right_index=True)

        all_errors['avg']= all_errors.iloc[:,2:].sum(axis=1)/9 #average error of all folds
        all_errors=all_errors.sort_values('avg')
        best=int(all_errors.iloc[0,0])
        

        train_MSE= all_errors.iloc[0,1]
        cv_results=all_errors.iloc[:,0:2]

#MAKE PREDICTIONS ON TESTS SET BASED ON THE BEST K WE JUST OBTAINED:
        
    if window_s > len(x_test.iloc[:,0]) or window_s+horizon > len(x_test.iloc[:,0]):
        print('The window size or horizon you chose is out of bounds')
        
    else:
        print('Phase 2: forecasting energy consumption on test set')
        cumulative_x_train= pd.DataFrame(x_test.iloc[:window_s,:], copy=True)            
        cumulative_y_train= pd.DataFrame(y_test.iloc[:window_s,:], copy=True) 
        
        to_predict_x= pd.DataFrame(x_test.iloc[window_s:window_s+horizon,:], copy=True) 
        to_predict_y= pd.DataFrame(y_test.iloc[window_s:window_s+horizon,:], copy=True) 
        y_hat= pd.DataFrame(np.zeros(shape=(horizon,1)), index=np.arange(1,horizon+1),columns=['y_hat'])
        
        for n in range(0,len( to_predict_x.iloc[:,0])):
            
            #Calcualte distances between each observation in x_data and query:
            distances= pd.DataFrame()
            distances['distance_to_query']= euclidean(cumulative_x_train, to_predict_x.iloc[[n],:])
            distances= pd.concat( [cumulative_x_train, distances, cumulative_y_train], axis=1)
            data_with_distances=distances.sort_values('distance_to_query')# distances in descending order
    
            #Calculate prediction based on K-NN:
            if best > len(cumulative_x_train.iloc[:,0]) | best==0:
                print("Unable to Complete task, K is larger than the total number of observations in the training set")
            else:
                k_neighbors= pd.DataFrame(data_with_distances.iloc[0:best, :], copy=True) #pick KNN
                y_hat.loc[n+1,'y_hat']= sum(k_neighbors.iloc[0:best, -1]) / best #Find avg
                
            #here I update the dataset and shape it according to the sliding window
            cumulative_x_train= cumulative_x_train.append(to_predict_x.iloc[[n],:])
            cumulative_x_train= pd.DataFrame(cumulative_x_train.iloc[-window_s:, :], copy=True)
            cumulative_y_train= cumulative_y_train.append(to_predict_y.iloc[[n],:])
            cumulative_y_train= pd.DataFrame(cumulative_y_train.iloc[-window_s:, :], copy=True)
            
            
        test_MSE = np.sqrt(mean_squared_error(to_predict_y, y_hat))
        y_hat.index= to_predict_y.index
        y_hat.columns= to_predict_y.columns
        
        
    print("The execution has finished.")
    return cv_results, best, train_MSE, test_MSE, y_hat


#--------TEST-------
#cv_results, best, train_MSE, test_MSE, y_hat= fit_knn_forecasting(x_train,y_train, x_test, y_test, 100, 50)

#plot results: 
#myFmt = DateFormatter('%Y-%m-%d') 
#fig, ax = plt.subplots()
#ax.plot(y_hat, color='blue', marker='o', linewidth=1, markersize=2, label='prediction')
#ax.plot(to_predict_y, color='green', marker='o', linewidth=1, markersize=2, label='observed')
#ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
#ax.set(title="Energy consumption")
#ax.xaxis.set_major_formatter(myFmt) #The YYYY-MM-DD formta
#fig.autofmt_xdate()
#ax.grid()    
#plt.legend()
#--------    -------

    
def knn_predict_t(k, x, y, x_t, f='10min'):
               
    future_y_v='unsuccessful' #This will get updated if the function is successfully executed
    #Target/query - the last observation of the dataset that was provided is used
    #to pretict t+1:
    temp_x = pd.DataFrame(x, copy=True)
    temp_y = pd.DataFrame(y, copy=True)
       
    num_obs= len(temp_x.iloc[:,0])
    if k > num_obs or k < 1:
        print("You have provided invalid parameters (K is too small/big)")    
    else:
        x_query= pd.DataFrame(x_t, copy=True)
        #Create a small empty df for the future prediction:
        future_x_v, future_y_v =ld.create_next_step(temp_x, f)
        x_query.index=future_x_v.index
        
        #Calcualte distances between each observation in x_data and query:
        distances= pd.DataFrame()
        distances['distance_to_query']= euclidean(temp_x, x_query)
        distances= pd.concat( [temp_x, distances, temp_y], axis=1)
        data_with_distances=distances.sort_values('distance_to_query')# distances in descending order

        #Calculate prediction based on K-NN provided by the user:
        k_neighbors= pd.DataFrame(data_with_distances.iloc[0:k, :], copy=True) #pick KNN
        forecast= sum(k_neighbors.iloc[0:k, -1]) / k #Find avg
                
        future_y_v.iloc[0,0]=forecast
        future_y_v.columns= temp_y.columns
        
    print("The execution has finished.")
    return future_y_v

#pred= knn_predict_t(1, x_test.iloc[-2,:], y_test.iloc[-2,:], a, f='10min')



# =============================================================================
# 
# 
#                                   GAUSSIAN PROCESS
# 
# =============================================================================
#RADIAL BASIS FUNCTION KERNEL:
def k(a,b,l,s):
    #a corresponds to np array 1
    #b corresponds to np array 2
    #l and s correspond to the kernel parameters
    transposed_b=b.T
    matrix= pow(s,2)#first term of the formula that then will become a matrix
    #Second part of the formula:
    scnd_term=pow(a,2).sum(1)# <--change sum axes 
    trd_term=pow(b,2).sum(1)# <--change sum axes 
    scnd_term=scnd_term.reshape(-1,1)
    matrix_product=np.matmul(a,transposed_b)
    mul=2*matrix_product
    scnd_term= scnd_term+trd_term-mul
    #Multiply both parts of the formula:
    exp= np.exp((-1/2)/pow(l,2)*scnd_term)
    matrix= matrix*exp
    
    return matrix

def optimization(X_train, Y_train):
    #This optimization function and the function that implements "minimize" were implemented
    #based on a solution developed by Krasser, (2018) full reference in project report. 
    #That solution was modified and adapted to fit my problem's requirements. 
    def negativelog_likelihood(theta):
        K = k(X_train, X_train, l=theta[0], s=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    return negativelog_likelihood                   


def futurePred(x_prior, y_prior, x_posterior, l, s, n):

    length=len(x_prior)
    c_mat_1=k(x_prior,x_prior,l, s)
    c_mat_1=c_mat_1+n**2*np.eye(length)#add corresponding noise to matrix
    
    linear_equation=solve(c_mat_1,k(x_prior,x_posterior,l,s),assume_a='pos')
    t_linear_e=linear_equation.T
    mu_post=np.matmul(t_linear_e,y_prior)
    
    post_c_matrix=k(x_posterior, x_posterior, l, s)-np.matmul(t_linear_e,k(x_prior, x_posterior, l, s))
    
    return post_c_matrix, mu_post


def fit_gaussian_process_forecasting(x_train, y_train, x_test, y_test, b=[0.0001, 0.0001, 0.01], window_s=100, horizon=50):
    
    #The following are the variables returned by the function, they are declared at this
    #point so that I don't get an error in case a value is not assigned to them
    #They will be updated as the function is executed.
    training_MSE='unsuccessful'
    test_MSE='unsuccessful'
    test_y_hat='unsuccessful'
    l_1='unsuccessful'
    s_1='unsuccessful'
    n_1='unsuccessful'
    
    print(" ")
    print(" ")
    print("Phase 1")
#First I need to transform the input data from Pandas DataFrames to Numpy Arrays:
    tarining_columns =x_train.columns
    tarining_index   =y_train.index
    test_columns     =x_test.columns
    test_index       =y_test.index
    
    np_x_train= np.array(x_train, copy=True)
    np_y_train= np.array(y_train, copy=True)
    np_x_test = np.array(x_test , copy=True)
    np_y_test = np.array(y_test , copy=True)
#Based on the assumption that we are dealing withy noisy data, I have to add a noise term
#to my data, it will be an initial guess but the noise term will be estimated later
#using an optimization function.     
    
    print("")
    print("")
    print("The optimization process has started")
    print("")
    print("")
    
    best_par= pd.DataFrame([[0,0, np.array([0])]], columns=['init_p', 'func_out', 'parameters'])
    
    initial_par =np.arange(0.1, 10.0, 0.5)    
    for each in range(len(initial_par)):

        try:
            #please read comment in the optimization function (above) to get more information 
            #about the following three lines, full reference in project report:
            
            output = minimize(optimization(np_x_train[0:window_s,:], np_y_train[0:window_s,:]), [initial_par[each],initial_par[each], initial_par[each]], 
                   bounds=((b[0], None), (b[1], None), (b[2], None)),
                   method='L-BFGS-B')
            #The third value in initial_par corresponds to the noise estimation.
            #output.fun    <-- objetive
            #output.x      <-- kernel parameters and noise
            
            best_par= best_par.append(pd.DataFrame([[initial_par[each] ,output.fun, output.x]], \
                                                   columns=['init_p', 'func_out', 'parameters']), ignore_index= True)
        except:
            print("")
            print("SVD did not converge in Linear Least Squares, a replacement value has been inserted in row: "+ str(each))
            print("Do not abort the execution.")
            print(str(each) + ' of '+ str(len(initial_par)))
            best_par= best_par.append(pd.DataFrame([[initial_par[each],10,]], columns=['init_p', 'func_out']), ignore_index= True)
            
    print("")
    print("")
    print("The optimization process has finished")
    best_par= best_par.drop(best_par.iloc[0,:])
    best_par= best_par.reset_index()
    best_par= best_par.iloc[:,1:]
    best_par= best_par.sort_values('func_out')
    best_par= best_par.dropna()
    
    

    if len(best_par) == 0 or (window_s+horizon) > len(np_y_train):
        print("Bounds need to be manually changed, update the bounds argument passed to the function")
        print("Or the window size and horizon are larger than your training dataset")
    else:
        optimized_par= best_par.head(1)
        l_1= optimized_par.iloc[0,2][0]
        s_1= optimized_par.iloc[0,2][1]
        print('Kernel parameters: '+ str(l_1)+ '   '+ str(s_1) )
        n_1= optimized_par.iloc[0,2][2]
        print('Noise: '+ str(n_1))
        
        print('Making predictions on training set')
        #The datasets that are utilizedn in the following line are adapted according to
        #the window size and horizon argumnets that were passed to the function
        #If the sum of those two arguments is > than the total number of obs. in the
        #training set then the function is terminated and a message is shown to the user
           
        M,  training_y_hat=futurePred(np_x_train[0:window_s,:], np_y_train[0:window_s,:], np_x_train[window_s:window_s+horizon,:], l_1, s_1, n_1)
        training_MSE = np.sqrt(mean_squared_error( np_y_train[window_s:window_s+horizon,:], training_y_hat))
              
        
        if (window_s+horizon) > len(np_y_test):
            print("The window size and horizon are larger than your TEST dataset")
        else:                               
            print("")
            print("Phase 2")
            print("Making predictions on test set ")
        
            M_2,  test_y_hat=futurePred(np_x_test[0:window_s,:], np_y_test[0:window_s,:], np_x_test[window_s:window_s+horizon,:], l_1, s_1, n_1)
            
            #Assign the corresconing date index to the test_y_hat:
            test_y_hat= pd.DataFrame(test_y_hat, index= y_test.iloc[window_s:window_s+horizon,:].index, columns=y_test.iloc[window_s:window_s+horizon,:].columns)
            
            test_MSE = np.sqrt(mean_squared_error( np_y_test[window_s:window_s+horizon,:], test_y_hat))
            
            print("The function has finished")
        

    return [l_1, s_1, n_1], training_MSE,  test_MSE, test_y_hat


#------GP TEST -----------
#a, gp_train_mse, gp_test_mse, test_y_hat= fit_gaussian_process_forecasting(x_train, y_train, x_test, y_test, [0.0001, 0.0001, 0.01], 100, 20)
#myFmt = DateFormatter('%Y-%m-%d') 
#fig, ax = plt.subplots()
#ax.plot(test_y_hat, color='blue', marker='o', linewidth=1, markersize=2, label='prediction')
#ax.plot(y_test.iloc[window_s:window_s+horizon,:], color='green', marker='o', linewidth=1, markersize=2, label='observed')
#ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
#ax.set(title="Energy consumption")
#ax.xaxis.set_major_formatter(myFmt) 
#fig.autofmt_xdate()
#ax.grid()    
#plt.legend()    
#--------         ----------------

def gaussian_process_predict(x_set, y_set, x_to_pred,  f='10min', pramaeters=[0.1, 0.1, 0.1]):
    
    prediction="No output"
    x_query= np.array(x_to_pred, copy=True)
    #Create a small empty df for the future prediction:
    future_x_v, future_y_v =ld.create_next_step(x_set, f)
    
    y_set_columns= y_set.columns
    x_set=np.array(x_set)
    y_set=np.array(y_set)
    y_set=y_set + pramaeters[2] * rd(1)
    
    try:
        other,  prediction=futurePred(x_set, y_set, x_query, pramaeters[0], pramaeters[1], pramaeters[2])
        
        prediction= prediction[0][0] #extract value from the output
        prediction= pd.DataFrame([prediction], columns=y_set_columns, index=future_y_v.index)
        
        print("Prediction for " + str(future_y_v.index[0]) + "  = "+ str(prediction.iloc[0,0])) 
        
    except:        
        print("The function was unable to produce an output, please veryfy the input you passed to it")
        
    return prediction



# =============================================================================
# 
#                           SUPPORT VECTOR REGRESSION
# 
# =============================================================================
# =============================================================================
# 

def train_svm(x_train, y_train, x_test, y_test, window_s=300, horizon=15):
    #This model was implemented based on the documentation provided by Scikit-learn.org (2019)
    #For full reference please refer to the project report. 
    
    #The following variables will get updated as the algorithm is executed
    search_result='unsuccessful'
    training_mse='unsuccessful'
    test_mse='unsuccessful'
    test_pred='unsuccessful'
    
    check=0 #if Gris search is successfull the algorithm continues
    
    if (window_s + horizon) > len(x_train.iloc[:,0]):
        print("Your window size + horizon are greater than your training set, the execution has stopped.")
    
    else:
        print("")
        print('GridSearch has started')
        print("")
        #hyper-parameters for grid search: 
        c_param=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        g_param=[0.0001,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.5, 3.0,4.0,5.0,7.0,10.0,15.0]
    
        try:
            search= GridSearchCV(SVR(kernel='rbf'), param_grid={"C": c_param, "gamma": g_param})
            search_result= search.fit(np.array(x_train.iloc[0:window_s,:]), np.array(y_train.iloc[0:window_s,:]).ravel())
            check=1
            
            #search.best_params_        <-- these are the optimal parameters returned by the function 
        except: 
            print("Grid Search has failed ")
            
        
        if check ==1:
            
            print("")
            print('Making predictions on training set')
            print("")
            training_pred= search_result.predict(np.array(x_train.iloc[window_s:window_s+horizon,:]))
            training_pred= pd.DataFrame(training_pred, index=x_train.iloc[window_s:window_s+horizon,:].index)
            training_mse=  np.sqrt(mean_squared_error(y_train.iloc[window_s:window_s+horizon,:], training_pred))
            
            #---------- inspection ----------
            #myFmt = DateFormatter('%Y-%m-%d') 
            #fig, ax = plt.subplots()
            #ax.plot(training_pred, color='blue', marker='o', linewidth=1, markersize=2, label='prediction')
            #ax.plot(y_train.iloc[window_s:window_s+horizon,:], color='green', marker='o', linewidth=1, markersize=2, label='observed')
            #ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
            #ax.set(title="Energy consumption")
            #ax.xaxis.set_major_formatter(myFmt) 
            #fig.autofmt_xdate()
            #ax.grid()    
            #plt.legend()   
            #-------------             ---------
    
    
            if (window_s + horizon) > len(x_test.iloc[:,0]):
                print("Your window size + horizon are greater than your test set, the execution has stopped.")
            else:
                print("")
                print('Making predictions on test set')
                print("")
                search_result= search.fit(np.array(x_test.iloc[0:window_s,:]), np.array(y_test.iloc[0:window_s,:]).ravel())
                test_pred= search_result.predict(np.array(x_test.iloc[window_s:window_s+horizon,:]))
                test_pred= pd.DataFrame(test_pred, index=y_test.iloc[window_s:window_s+horizon,:].index, columns= y_test.columns)
                test_mse=  np.sqrt(mean_squared_error(y_test.iloc[window_s:window_s+horizon,:], test_pred))
                      
                print("End of the execution")

    return search_result, training_mse, test_mse, test_pred
##search.best_params_
    

#---- TEST --------
#search_result, svr_training_mse, svr_test_mse, svr_test_pred=train_svm(x_train, y_train, x_test, y_test, window_s=400, horizon=15)

#---------- inspection ----------
#myFmt = DateFormatter('%Y-%m-%d') 
#fig, ax = plt.subplots()
#ax.plot(svr_test_pred, color='blue', marker='o', linewidth=1, markersize=2, label='prediction')
#ax.plot(y_test.iloc[window_s:window_s+horizon,:], color='green', marker='o', linewidth=1, markersize=2, label='observed')
#ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
#ax.set(title="Energy consumption")
#ax.xaxis.set_major_formatter(myFmt) 
#fig.autofmt_xdate()
#ax.grid()    
#plt.legend()           
#-------------             ---------
#-------------             ---------


def predict_svm(search_result, x_data, to_predict, f='10min'):
    
    prediction= search_result.predict(np.array(to_predict))
    future_x_v, future_y_v =ld.create_next_step(x_data, f)
    future_y_v.iloc[0,0]=prediction
    
    return future_y_v
    
  
