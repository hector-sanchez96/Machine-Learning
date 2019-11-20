# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:01:34 2018

@author: Héctor Sánchez
"""
import os
import pyodbc #this is to connect to sql server 
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize
import pandas as pd 
import math
import numpy as np
import nltk # to tokenize my strings
os.chdir('//urbanscience.net/users/ASC/hsanchez/Documents/ML/tf–idf')
from scipy import spatial # for cosine similarity


from data_retrieval_and_cleansing import * 

#At  this point the raw data are in their corresponding variables and I can start
#preparing them for tf-idf:


#Dictionry:
#data 1 is historic and data 2 is to be segmented
#data1 is = s_pc_historic (all records)
#data 2 is = s_pc_to_segment (all recors)
def dictionary(data1, data2):
    
    my_set= set()
    for each in range(len(data1)):
        d=data1.iloc[each]['complete_name'].split()
        for i in range(len(d)):
            my_set.add(d[i])
    
    for each in range(len(data2)):
        d=data2.iloc[each]['complete_name'].split()
        for i in range(len(d)):
            my_set.add(d[i])
    my_set=list(my_set)
    return my_set



#TF ------
#DOC must be a single line from s_pc_to_segment
# to be passed only the one row in doc and the entire dictionary
def tf (doc, dictionary):
    doc=doc.loc[0, 'complete_name']
    doc=doc.split()
    tf=np.zeros(len(dictionary))
    counts={}
    
    for each in range(len(dictionary)):
        if dictionary[each] in doc:
            c=0    
            for i in range(len(doc)):
                if doc[i]== dictionary[each]:
                    c +=1
            tf[each]=c
        
    count=0
    for each in range(len(tf)):
        count =count +tf[each]
        
    tf= tf/count#normalization
    return tf

                
#IDF - - - - - - 
#Pass the entire dictionary
#doc is = all the documents in the corpus
def idf(doc, s_x_historic, dictionary):

    s_x_historic=s_x_historic.append(doc, ignore_index= True)
    m= np.zeros((len(dictionary)))
    doc=doc.loc[0,'complete_name']
    doc=doc.split()

    for each in range(len(dictionary)):
        
        sum_of_terms=0
        if dictionary[each] in doc:
            
            for i in range(len(s_x_historic)):
                
                historic_models=s_x_historic.loc[i, 'complete_name']
                historic_models=historic_models.split()
                for j in range(len(historic_models)):
                    if historic_models[j] == dictionary[each]:
                        sum_of_terms +=1
        m[each]=sum_of_terms
        
    for x in range(len(m)):
        if m[x]>0:
            m[x]=math.log10(len(s_x_historic)/1+m[x])
    return m




#RANKING
#Matrix is the distance matrix among all documents
#original_to_segment is = pc_to_segment
#original_dataframe is = pc_historic
#docu is = all documents in the corpus
#final_ranking=pd.DataFrame()
def ranking(matrix, original_to_segment, original_dataframe, docu):
    rank= pd.DataFrame()
    for k in range(1,len(matrix.columns)):
        to_segment=original_to_segment.loc[original_to_segment['unique_id'] == docu.loc[0]['unique_id']][['unique_id', 'Make', 'Model']].copy()
        to_segment=to_segment.reset_index()
        to_segment=to_segment.drop(columns=['index'])
        
        
        suggestion=original_dataframe.loc[original_dataframe['unique_id'] == docu.loc[k]['unique_id']][['unique_id', 'Make', 'Model','SegmentId']].copy()
        suggestion['similarity']=distances_matrix.loc[0][k]
        suggestion=suggestion.reset_index()
        suggestion=suggestion.drop(columns=['index'])
        
        line = pd.concat([to_segment, suggestion], axis=1, sort=False)
        rank= rank.append(line)
    return rank


#from data_retrieval_and_cleansing import * 

#Change filters accordingly !!! 
#from data_retrieval_and_cleansing import * 

pc_to_segment=pc_to_segment.loc[[48, 97, 50,101]].copy()
pc_to_segment=pc_to_segment.reset_index()
pc_to_segment=pc_to_segment.drop(columns=['index'])
s_pc_to_segment=s_pc_to_segment.loc[[48, 97, 50, 101]].copy()
s_pc_to_segment=s_pc_to_segment.reset_index()
s_pc_to_segment=s_pc_to_segment.drop(columns=['index'])

pc_historic=pc_historic.loc[[1464, 365, 1331, 19, 200, 69]]
pc_historic=pc_historic.reset_index()
pc_historic=pc_historic.drop(columns=['index'])
s_pc_historic=s_pc_historic.loc[[1464, 365, 1331, 19, 200, 69]]
s_pc_historic=s_pc_historic.reset_index()
s_pc_historic=s_pc_historic.drop(columns=['index'])


final_ranking = pd.DataFrame()


s_pc_historic=s_pc_historic.reset_index()
s_pc_historic=s_pc_historic.drop(columns=['index'])

pc_historic=pc_historic.reset_index()
pc_historic=pc_historic.drop(columns=['index'])


original_to_segment=s_pc_historic.copy()
original_pc_historic=pc_historic.copy()


#pc_dictionary= dictionary(s_pc_historic, s_pc_to_segment)


for each in range(len(s_pc_to_segment)):
    
    s_pc_historic=original_to_segment.copy()
    pc_historic=original_pc_historic.copy()
    
    s_pc_historic=s_pc_historic.reset_index()
    s_pc_historic=s_pc_historic.drop(columns=['index'])
    
    pc_historic=pc_historic.reset_index()
    pc_historic=pc_historic.drop(columns=['index'])
    
    if pc_to_segment.loc[each]['Model']== ' ' or pc_to_segment.loc[each]['Model']== '':
        print('Model is empty for vehicle'+ str(each+1)+ ' !!!!')
        continue
     
    relevant_historic=set()
    
    string=pc_to_segment.loc[each]['Make'].split()
    for p in range(len(pc_historic)):
        a=pc_historic[pc_historic['Make'].str.contains(string[0])]
        b=a.index.tolist()
        
        for m in range(len(b)):
            relevant_historic.add(b[m])     

    relevant_historic=list(relevant_historic)
    
    if len(relevant_historic) < 1:
        print('document ' + str(each+1) + ' NOT FOUND')
        continue
    
    s_pc_historic=s_pc_historic.ix[relevant_historic]
    s_pc_historic=s_pc_historic.reset_index()
    s_pc_historic=s_pc_historic.drop(columns=['index'])
    
    pc_historic=pc_historic.ix[relevant_historic]
    pc_historic=pc_historic.reset_index()
    pc_historic=pc_historic.drop(columns=['index'])
    
    
    pc_dictionary= dictionary(s_pc_historic, s_pc_to_segment)
    
    
    doc=s_pc_to_segment.loc[[each]][:].copy() 
    documents=doc.append(s_pc_historic, ignore_index= True)
    distances_matrix = pd.DataFrame(index=range(len(documents)), columns=range(len(documents)))
    distances_matrix = distances_matrix.fillna(0) # with 0s rather than NaNs

    for j in range(len(documents)):
        single_doc_a=documents.iloc[[j], :].copy()
        single_doc_a=single_doc_a.reset_index()
        single_doc_a=single_doc_a.drop(columns=['index'])
        
        tfidf1=tf(single_doc_a, pc_dictionary) * idf(single_doc_a, s_pc_historic, pc_dictionary)
        
        for i in range(len(documents)):
            single_doc_b=documents.iloc[[i], :].copy()
            single_doc_b=single_doc_b.reset_index()
            single_doc_b=single_doc_b.drop(columns=['index'])
            
            tfidf2=tf(single_doc_b, pc_dictionary) * idf(single_doc_b, s_pc_historic, pc_dictionary)
            
            distance=1 - spatial.distance.cosine(tfidf1, tfidf2)
            
            distances_matrix.loc[[j], i]=distance
    
    final_ranking =final_ranking.append(ranking(distances_matrix, pc_to_segment, pc_historic, documents))
    print('Processed documents: ' + str(each+1) + ' Total documents: ' + str(len(s_pc_to_segment)))

