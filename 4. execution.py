# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 02:02:34 2019

@author: hsanchez
"""


#PC SECTION: 
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

pc_to_segment=pc_to_segment.loc[[48, 97, 50]].copy()
pc_to_segment=pc_to_segment.reset_index()
pc_to_segment=pc_to_segment.drop(columns=['index'])
s_pc_to_segment=s_pc_to_segment.loc[[48, 97, 50]].copy()
s_pc_to_segment=s_pc_to_segment.reset_index()
s_pc_to_segment=s_pc_to_segment.drop(columns=['index'])

pc_historic=pc_historic.loc[[1464, 365, 1331, 19, 200, 69]]
pc_historic=pc_historic.reset_index()
pc_historic=pc_historic.drop(columns=['index'])
s_pc_historic=s_pc_historic.loc[[1464, 365, 1331, 19, 200, 69]]
s_pc_historic=s_pc_historic.reset_index()
s_pc_historic=s_pc_historic.drop(columns=['index'])

pc_dictionary= dictionary(s_pc_historic, s_pc_to_segment)

final_ranking = pd.DataFrame()
for each in range(len(s_pc_to_segment)):
    
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
    
