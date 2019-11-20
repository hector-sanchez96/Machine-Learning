
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler


def data_inputation(null_index, null_data, n_column, col_name):
    #The purpose of this function is to replace missing values in non-categorical columns.
    
    if len(list(null_index))>0:#exectue this only if there is one or more nulls
        
        for this_one in null_index:
            avg=1#in case the try catch block does not work
            c_type=1#if this variable is 1 it means that I can calculate an average with the data that I have, given that it has the right values (numerical)
            #if the exception below changes it to 0 then I cannot continue with the data inputation and the user must fix the non-numerical values
            
            
            day= null_data.loc[this_one, 'day']
            month= null_data.loc[this_one, 'month']
            
            #if the null value is in day 1 and month 1 relevant data contains non-null values
            #in day 1 and month 1 that could be used to calculate and average
            relevant_data= pd.DataFrame(n_column.loc[ ( n_column['day'] == day)  & (n_column['month'] == month) & ( n_column[col_name] != 'Null')  & (n_column[col_name] != 'null') & ( n_column[col_name] != 'NaN') & ( n_column[col_name] != '') & ( n_column[col_name] != 'nan')] , copy=True)
            
            try:
                relevant_data[col_name]= pd.to_numeric(relevant_data[col_name])
            except:
                c_type=0
                print("A column that is supposed to have only numerical values (not categorical) seems to also have non-numerical values")
                print("Please make sure that all the values in the numerical columns exclude any type of non-numerical characters")
                print("Once that is done, please run the function again")
                
            
            
            #this is to check that the column values have the right data type 
            if avg==0 and c_type==1:
                try:
                    avg= relevant_data[col_name].mean()
                except:
                    print("Please check the values in "+ col_name + ", the values don't have the expected format (might contain letters or invalid characters)")
                    
                    
            #second attempt: if the criteria in relevant data does not work, I change it and try again
            #If this does not work, then you need to check the values that you are provided
            if c_type==0:
                try:
                    relevant_data= pd.DataFrame(n_column.loc[ ( n_column[col_name] != 'Null')  & (n_column[col_name] != 'null') & ( n_column[col_name] != 'NaN') & ( n_column[col_name] != '') & ( n_column[col_name] != 'nan')] , copy=True)
                    relevant_data[col_name]= pd.to_numeric(relevant_data[col_name])
                    avg= relevant_data[col_name].mean()
                except:
                    print("It was not possible to substitue missing values, please check the format of the values in the columns that are supposed to be numerical (non-categorical)")
                    print("for the moment, 1s have been assigned")
                
            
            n_column.loc[this_one, col_name]= avg
    
        
        try:
            n_column[col_name]= pd.to_numeric(n_column[col_name])
        except:
            print("A column that is supposed to have only numerical values (not categorical) seems to also have non-numerical values")
            print("Please make sure that all the values in the numerical columns exclude any type of non-numerical characters")
            print("Once that is done, please run the function again")
            
        
    #n_column.info(memory_usage='deep')
    return n_column[col_name]

 

def load( file, categorical=[]):
    
    data_ok= 0 #to check if a valid location (file path) has been provided
    
    #I store the raw data here
    temporal_df= pd.DataFrame()

    categorical=np.array(categorical)
    #index of columns starts from 0 that is why I substract one from the input:
    categorical= categorical-1
        
    #Global varables    
    category_dict={}
    category_counter=0    
    
    #-----LOAD RAW DATA:
    try:
        raw_data = pd.read_csv(file)
        data_ok=1
    except:
        print("The file path you provided is invalid or the file is not stored in the location you specified")
    
    
    
    if data_ok == 1:
        
        raw_data=raw_data.drop_duplicates(subset=[raw_data.columns[0]], keep='first')#remove duplicated time stamps
        
        num_attributes=len(raw_data.columns)
        
        #I change the data type of each column so that later on I can make comparisons bteween same-type values
        for i in range(num_attributes):
            if i == 0:
                continue
            else:
                 raw_data.iloc[:, i] =  raw_data.iloc[:, i].astype(str)
        
            
        #Date:
        temporal_df['date']= pd.to_datetime(raw_data.iloc[:,0], format='%d/%m/%Y %H:%M')
        
        
        #Derived attributes from date attribute:
        #Day:
        temporal_df['day']=temporal_df['date'].dt.dayofweek
        temporal_df['day']=temporal_df.loc[:,'day'].astype(np.int8)
        
        #Month:
        temporal_df['month']=temporal_df['date'].dt.month
        temporal_df['month']=temporal_df.loc[:,'month'].astype(np.int8)
        
        #Season:
        temporal_df['season']=(temporal_df['month']%12+3)//3
        temporal_df['season']=temporal_df.loc[:,'season'].astype(np.int8)
        
        
        #if the number of attributes in the data set is more than 2 (date and energy consumption)
        #I manually go through each column and add it to the dataset 
        #if one of the columns is catagorical and it was flagged as categorical
        #when this function was invoked, I convert the categorical values (as strings)
        #into numbers. If the column is not categorical then I simply add it to 
        #the data frame after replacing missing values. 
        if num_attributes > 2:
            
            column_names=list(raw_data.columns)
            
            for each in np.arange(num_attributes):
                
                if each != 0:
                    
                    col_name=column_names[each]
                    #n_column stores the current column and is used to find and replace nulls
                    n_column=pd.DataFrame(temporal_df.loc[:,['date', 'day', 'month', 'season']].copy())
                    n_column[col_name]= pd.DataFrame(raw_data.iloc[:,each].copy())
                    
                    null_data=pd.DataFrame(n_column.loc[ ( n_column.loc[:,col_name] == 'Null')  | (n_column.loc[:,col_name] == 'null') | ( n_column.loc[:,col_name] == 'NaN') | ( n_column.loc[:,col_name] == '') | ( n_column.loc[:,col_name] == 'nan')] , copy=True)
                    null_index=null_data.index.values
                    
                    
                    #Test after function is executed to check if there is any null in x column
                    #b= list(temporal_df.loc[:,col_name])
                    #'Null' in b
                    
                        
                    if each in categorical:#if column x in flagged categorical columns:
                        
                        #Categorical values:
                        #The following 2 loops create a lookup table that is used to replace all the 
                        #categorical values with a number that represents its category:
                        
                        temporal_column=raw_data.iloc[:,each].copy()
                        unique_categories_in_column= list(temporal_column.unique())
                        curren_keys=list(category_dict.keys())
                        for q in unique_categories_in_column:
                            if q not in curren_keys:
                                category_dict.update({q:category_counter})
                                category_counter +=1
                        
                        curren_keys=list(category_dict.keys())        
                    
                        #Once I have the categories codifed, I assign each code to the corresponding category
                        temporal_column= pd.DataFrame(temporal_column)
                        for x in range(len(unique_categories_in_column)):
                            temporal_column.loc[temporal_column.iloc[:,0] == unique_categories_in_column[x]]= category_dict.get(unique_categories_in_column[x])
                        
        
                        
                        temporal_df[col_name]= temporal_column
                        
                        
        # =============================================================================
        # Here I load non-categorical columns and perform data inputation if necessary:
                    elif len(list(null_index))>0:
                        temporal_df[col_name]=raw_data.iloc[:,-1]
                        temporal_df[col_name]=data_inputation(null_index, null_data, n_column, col_name)
                         
                    else:
                        #If the column is not categorical and has no missing values:
                        temporal_df[col_name]=raw_data.iloc[:,each].copy()
        # =============================================================================
                        
                    category_dict={}
                    category_counter=0
                    
                    
                    
                    
        
        #If the number of attributes is NOT > 2 --> just time stamp and energy consumption are provided    
        for k in np.arange(num_attributes):
            if k != 0:
                #when only date and energy consumption are given in the raw data set:
                if num_attributes == 2:
                    
                    column_names=list(raw_data.columns)
                    col_name=column_names[k]
                    #n_column stores the current column and is used to find and replace nulls
                    n_column=pd.DataFrame(temporal_df.loc[:,['date', 'day', 'month', 'season']].copy())
                    n_column[col_name]= pd.DataFrame(raw_data.iloc[:,k].copy())
                    
                    null_data=pd.DataFrame(n_column.loc[ ( n_column.loc[:,col_name] == 'Null')  | (n_column.loc[:,col_name] == 'null') | ( n_column.loc[:,col_name] == 'NaN') | ( n_column.loc[:,col_name] == '') | ( n_column.loc[:,col_name] == 'nan')] , copy=True)
                    null_index=null_data.index.values
                    
                    if len(list(null_index))>0:
                                temporal_df[col_name]=raw_data.iloc[:,-1]
                                temporal_df[col_name]=data_inputation(null_index, null_data, n_column, col_name)
                                 
                    else:
                        temporal_df[col_name]=raw_data.iloc[:,k].copy()
            
    
        responses= pd.DataFrame()        
        responses['energy_consumption']= temporal_df.iloc[:, -1] 
        
        temporal_df= temporal_df.drop(temporal_df.columns[-1], axis=1)
        temporal_df= pd.concat( [temporal_df, responses], axis=1)         
        
        
        #I change the data type of each column so that later on I can make comparisons bteween same-type values
        for j in range(len(temporal_df.columns)):
            if j == 0:
                continue
            else:
                try:
                    temporal_df.iloc[:, j]= pd.to_numeric(temporal_df.iloc[:, j])
                except:
                    print("A column that is supposed to have only numerical values (not categorical) seems to also have non-numerical values")
                    print("Please make sure that all the values in the numerical columns exclude any type of non-numerical characters")
                    print("Once that is corrected, please run the function again")
                 
        temporal_df.index = temporal_df['date']
        temporal_df= temporal_df.drop(temporal_df.columns[0], axis=1)#drop old datestamp column which has become the new index
        temporal_df= temporal_df.drop(temporal_df.columns[[0,1,2]], axis=1)#drop day, month, season
        print('The function has terminated, please continue to see the data you have loaded')                          
    
    return temporal_df
#temporal_df.info(memory_usage='deep')

#data_df=pd.DataFrame(new_data.iloc[:10, :], copy=True)    
def shift_data(data_df, lag_num=1):
    ind_column= data_df.index
    
    if lag_num < 1:
        print('The number of lags you requested is invalid')
    else:
        
        final_lag =pd.DataFrame(index=ind_column)#this df will contain all lagged attributes
        future_x  = pd.DataFrame(data_df.iloc[[-1],:])#lag t-1 to predict t
        lags_for_future_x= pd.DataFrame(index=future_x.index)
        
        #This loop is to crreate the lags for the entire dataset
        for each in range(lag_num):
            
            
            temp_lag=data_df.shift(each+1)
            
            for i in range(len(temp_lag.columns)):
                #I have this loop because each lagged attribute needs a different name
                temp_lag.rename(columns={(str(temp_lag.columns[i])): (str(temp_lag.columns[i]) +'-' +str(each+1))}, inplace=True)                
            
            #Merge all laggs (1...n) in one df
            final_lag=final_lag.merge(temp_lag, left_index=True, right_index=True)
            
        
        #redefine column names for first lag in attributes for future prediction
        for m in range(len(future_x.columns)):
                #I have this loop because each lagged attribute needs a different name
                future_x.rename(columns={(str(future_x.columns[m])): (str(future_x.columns[m]) +'-' +str(1))}, inplace=True)
        
        if lag_num>1:
            
            #This loop is to crreate the lags for the last observation which will be utilized to predict t
            for n in range(lag_num-1):
                
                
                temp_lag_for_x=data_df.shift(n+1)
                
                for j in range(len(temp_lag_for_x.columns)):
                    #I have this loop because each lagged attribute needs a different name
                    temp_lag_for_x.rename(columns={(str(temp_lag_for_x.columns[j])): (str(temp_lag_for_x.columns[j]) +'-' +str(n+2))}, inplace=True)                
                
                #Merge all laggs (1...n) in one df
                lags_for_future_x=lags_for_future_x.merge(temp_lag_for_x, left_index=True, right_index=True)
           
        future_x= future_x.merge(lags_for_future_x, left_index=True, right_index=True)
        future_x= np.array(future_x.values)
        future_x= pd.DataFrame(future_x, columns=final_lag.columns)
        

        #Merge original dataset (Y) with its corresponding lags 
        data_df= pd.DataFrame(data_df.iloc[:,-1])# this refers to the last column which is Y
        data_df= final_lag.merge(data_df, left_index=True, right_index=True)
        data_df= data_df.dropna()
        
    return data_df, future_x
#function testing:
#data_df= pd.DataFrame(data, copy=True)
#b= shift_data(data, 3)
    


def split_data(data, split_percentage=75):

    split_percentage= int(split_percentage)
    #upper_limit is utilized to define how many rows the training set should contain
    upper_limit= (split_percentage * len(data.iloc[:,0]))/100
    upper_limit= int(upper_limit)
    
    x_train= pd.DataFrame(data.iloc[0:upper_limit, 0:-1]) 
    y_train= pd.DataFrame(data.iloc[0:upper_limit, -1])
    
    x_test=  pd.DataFrame(data.iloc[upper_limit:, 0:-1])
    y_test=  pd.DataFrame(data.iloc[upper_limit:, -1])
    
    return x_train, y_train, x_test, y_test



def create_next_step(data_f_x, fre='10min'):
    
    
    last_t     = data_f_x.iloc[[-1],:] #last observation in dataset
    last_index =  last_t.index       #last obesrvation's index
    last_index = str(last_index[0])  
    col        = len(last_t.columns) #number of columns
    
    new_index = pd.DataFrame(pd.date_range(start=last_index, periods=2, freq=fre))
    new_index = str(new_index.iloc[1,0])
    
    x = pd.DataFrame(np.zeros(shape=(1,col)))
    x.index=[pd.to_datetime(new_index)]
    
    y = pd.DataFrame(np.zeros(shape=(1,1)))
    y.index=[pd.to_datetime(new_index)]
    y.columns=['energy_consumption']
    
    return x, y
#a,b=create_next_step(x_test, '30min')



def scale_data(new_data):
#	This standardization methodology was proposed by Brownlee, J. (2016) --> Full reference in report
#   Certain modifications were made to adapt the concept to my specific case. 
    columns=new_data.columns#preserve column names 
    ind= new_data.index
    
    
    new_df= pd.DataFrame()
        
    s = MinMaxScaler().fit(new_data)
    transformed=s.transform(new_data)
    new_df= transformed
    new_df= pd.DataFrame(new_df, index=ind, columns=columns)
    
    return s, new_df

#function test:
#s, a= scale_data(new_data)

def descale_data(scale, new_data):
#	This standardization methodology was proposed by Brownlee, J. (2016) --> Full reference in report
#   Certain modifications were made to adapt the concept to my specific case. 
    columns=new_data.columns#preserve column names 
    ind= new_data.index
    
    new_df= pd.DataFrame()

    transformed=scale.inverse_transform(new_data)
    new_df= transformed
    new_df= pd.DataFrame(new_df, index=ind, columns=columns)
    
    return new_df
#function test:
#b=descale_data(s,a)


def ts_decomposition(df, frequency, mod='multiplicative'):
    #this function returns a time series decomposition, the data you pass to it
    #must be a pandas DF with its corresponding date index
    #the frequency must be an integer and the model can be additive or multiplicative
    decomposition = seasonal_decompose(pd.DataFrame(df.iloc[:,-1]),
                                       freq=frequency,
                                       model=mod)
    trend=decomposition.trend
    seasonal=decomposition.seasonal
    residual=decomposition.resid
    
    #return new data frame without trend and seasonal components:
    df.iloc[:,-1]= residual
    df= df.dropna()
    
    return trend, seasonal, residual, df, decomposition
 


#FOR TESTING PURPOSES:
# =============================================================================    
#for testing purposes:
#file=r"C:\Users\hecto\OneDrive\Documents\MSC\FINAL PROJECT\Development\Data\dataset_2_multivariate.csv"
#file=r"C:\Users\hecto\OneDrive\Documents\MSC\FINAL PROJECT\Development\Data\dataset_1.csv"


#data=load(file)
#data=data.iloc[-8000:,:]
#data=pd.DataFrame(data['energy_consumption'].diff()) 
#data= data.dropna()   

# #STEP BY STEP: 
# new_data=pd.DataFrame(data, copy=True)
#new_data= pd.DataFrame(new_data.iloc[:,[0,2,4]])
#new_data= pd.DataFrame(new_data.iloc[:,-1])
# trend, seasonal, residual, new_data, decomposition=ts_decomposition(new_data, 144)    
# s, new_data= scale_data(new_data)
# new_data, future_x=shift_data(new_data, 1)  
# x_train, y_train, x_test, y_test= split_data(new_data, 93)    
