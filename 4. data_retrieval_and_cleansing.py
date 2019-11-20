# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:49:41 AM 2018

@author: Héctor Sánchez
"""

import pyodbc #this is to connect to sql server 
import pandas as pd 
import math
import numpy as np

#Establish the connection 
cnxn = pyodbc.connect('Driver={SQL Server};'
                      'Server=ASCURBANSCI02\SQL2005;'
                      'Database=DataNissanBaltics;'
                      'Trusted_Connection=yes;')


# ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦ DATA EXTRACTION ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦ 
 
#Extract the data (models that need to be segmented): 
#sql="select SegmentId, Make, Model, EuroCode from usisegmentrules_test where segmentid is null"
sql="select SegmentId, Make, Model, EuroCode from usisegmentrules where segmentid is null"
df=pd.read_sql(sql, cnxn)#this dataframe contains all models that need to be segmented
df= pd.DataFrame(df, columns=['SegmentId', 'Make', 'Model', 'EuroCode'])
pc_to_segment=df[df['EuroCode']=='PC']#PC MODELS TO BE SEGMENTED
cv_to_segment=df[df['EuroCode']=='CV']#CV MODELS TO BE SEGMENTED


#Extract comparable models to do the segmentation: 
sql="select distinct ModelId_5 as SegmentId, Model as Make, ' ' as Model, car as EuroCode  from modelcrosstab_total where car ='car'"
pc_crosstab=pd.read_sql(sql, cnxn)
pc_crosstab= pd.DataFrame(pc_crosstab, columns=['SegmentId', 'Make', 'Model', 'EuroCode'])
sql="select distinct ModelId_5 as SegmentId, Model as Make, ' ' as Model, car as EuroCode  from modelcrosstab_total where car ='LCV'"
cv_crosstab=pd.read_sql(sql, cnxn)
cv_crosstab= pd.DataFrame(cv_crosstab, columns=['SegmentId', 'Make', 'Model', 'EuroCode'])
sql="select distinct SegmentId, Make, Model, EuroCode from usisegmentrules_test where segmentid is not null and eurocode ='pc'"
pc_historic=pd.read_sql(sql, cnxn)
pc_historic= pd.DataFrame(pc_historic, columns=['SegmentId', 'Make', 'Model', 'EuroCode'])
sql="select distinct SegmentId, Make, Model, EuroCode from usisegmentrules_test where segmentid is not null and eurocode ='cv'"
cv_historic=pd.read_sql(sql, cnxn)
cv_historic= pd.DataFrame(cv_historic, columns=['SegmentId', 'Make', 'Model', 'EuroCode'])


#Append "lookup tables" according to cv/pc models
pc_historic=pc_historic.append(pc_crosstab, sort=True, ignore_index=True) 
cv_historic=cv_historic.append(cv_crosstab, sort=True, ignore_index=True)

#Clear used variables:
del(sql, df,pc_crosstab,cv_crosstab)
# =============================================================================




#¦¦¦¦¦¦¦¦¦¦¦¦DATA CLEANSING¦¦¦¦¦¦¦¦¦¦¦¦
# =============================================================================
# #This section uses a function that does all the cleansing so it just has to 
# #be invoked.For any update, modify the function
#dataframe is the actual data and data_type defines if the data that is being passing in 
#is historic data or models to be segmented ->used for the if statement
# 1 represents data to be segmented and 2 is historic data




def clean (dataframe,data_type):
#     # Important to convert them to lowercase:
# =============================================================================
    dataframe.loc[:,'Make']=dataframe['Make'].str.lower()
    dataframe.loc[:,'Model']=dataframe['Model'].str.lower()
    dataframe.loc[:,'EuroCode']=dataframe['EuroCode'].str.lower()
# =============================================================================
#Remove unwanted characters:      
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( ')' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( '(' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( '_' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( ',' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( '-' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( '/' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( '.' , ' ')
    dataframe.loc[:,'Make'] = dataframe.loc[:,'Make'].str.replace( "'" , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( ')' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( '(' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( '_' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( ',' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( '-' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( '/' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( '.' , ' ')
    dataframe.loc[:,'Model'] = dataframe.loc[:,'Model'].str.replace( "'" , ' ')
# =============================================================================
#To have the same terminology among attributes
    dataframe.loc[dataframe['EuroCode']=='car','EuroCode']='pc'
    dataframe.loc[dataframe['EuroCode']=='lcv','EuroCode']='cv'
#To remove what I consider stop words:
    dataframe.loc[dataframe['Make']=='pc','Make']=' '
    dataframe.loc[dataframe['Model']=='cv','Model']=' '

# add incremental column that will help me to uniquely identify each record
    dataframe.insert(0, 'unique_id', range(0, 0 + len(dataframe)))
#Remove unnecessary codes in those
    if data_type == 2:
        dataframe.loc[:,'Make'] = dataframe['Make'].str.replace( 'cv' , ' ')
        dataframe.loc[:,'Make'] = dataframe['Make'].str.replace( 'pc' , ' ')
    
    return dataframe


clean(pc_historic,2)
clean(cv_historic,2)
clean(pc_to_segment,1)
clean(cv_to_segment,1)



s_pc_historic= pd.DataFrame( columns=['unique_id', 'complete_name'])
s_pc_historic['complete_name']=pc_historic['Make']+ ' '+ pc_historic['Model']
s_pc_historic['unique_id']=pc_historic['unique_id']
    
s_cv_historic= pd.DataFrame( columns=['unique_id', 'complete_name'])
s_cv_historic['complete_name']=cv_historic['Make']+ ' '+ cv_historic['Model']
s_cv_historic['unique_id']=cv_historic['unique_id']
    
s_pc_to_segment= pd.DataFrame( columns=['unique_id', 'complete_name'])
s_pc_to_segment['complete_name']=pc_to_segment['Make']+ ' '+ pc_to_segment['Model']
s_pc_to_segment['unique_id']=pc_to_segment['unique_id']
    
s_cv_to_segment= pd.DataFrame( columns=['unique_id', 'complete_name'])
s_cv_to_segment['complete_name']=cv_to_segment['Make']+ ' '+ cv_to_segment['Model']
s_cv_to_segment['unique_id']=cv_to_segment['unique_id']


# =============================================================================
    
# =============================================================================


#¦¦¦¦¦¦¦¦¦¦¦¦DATA FORMATTING¦¦¦¦¦¦¦¦¦¦¦¦
# =============================================================================
#The following dictionaries will store key-value pairs that contain previously segmented models
# =============================================================================
# pc_dictionary={}
# cv_dictionary={}
# cv_historic_list=[]
# pc_historic_list=[]
# =============================================================================
# =============================================================================
# 
# for index, row in pc_historic.iterrows():
#     segmentid= str(row['SegmentId'])
#     make= str(row['Make'])
#     model= str(row['Model'])
#     eurocode= str(row['EuroCode'])  
#     string= make+' '+model+' '+eurocode
#     
#     pc_dictionary[string]=segmentid
#     
#     if string not in pc_historic_list:
#         pc_historic_list.append(string)
#     
# for index, row in cv_historic.iterrows():
#     segmentid= str(row['SegmentId'])
#     make= str(row['Make'])
#     model= str(row['Model'])
#     eurocode= str(row['EuroCode'])  
#     string= make+' '+model+' '+eurocode
#     
#     cv_dictionary[string]=segmentid
#     
#     if string not in cv_historic_list:
#         cv_historic_list.append(string)
# 
# =============================================================================
