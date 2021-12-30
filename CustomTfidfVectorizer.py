#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: JonnyFLDN
"""


import numpy as np
import pandas as pd
from scipy.sparse.dia import dia_matrix
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency

class CustomTfidf(TfidfVectorizer):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        
        ''' Default sklearn's TfidVectorizer settings'''
        super(CustomTfidf, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype,use_idf =use_idf,smooth_idf=smooth_idf)
        
        #variables 
        self.docs = None
        self.doc_freq = None
        self.new_docs = [] 
    
    
    @staticmethod
    def add_dimensions(item):
        
        ''' Convert item into an iterable ''' 
        
        arr = np.asarray(item)
        item = arr if arr.ndim !=0 else arr.reshape(-1)
        return item
    
    def fit(self,docs):
        
        ''' Applies TfidVectorizer fit method and stores info for re-fit
        
        Parameters 
        ----------
        docs: iterable 
            contains str
        
        Returns 
        ----------
        self   
        
        ''' 
        X = super(CustomTfidf,self).fit(docs)
        self.docs = np.asarray(docs)
        #if use_idf = false then we transform and sum ? what about norm?
        self.doc_freq = _document_frequency(X.transform(docs)) #questionable
        return self
    
    def transform(self,docs):
        
        ''' Applies TfidVectorizer transform method
        
        Parameters 
        ----------
        docs: iterable 
            contains str
        
        Returns 
        ----------
        X : Document-term matrix, [n_samples, n_features]
        
        '''
        
        X= super(CustomTfidf,self).transform(docs)
        return X
    
    def fit_transform(self,docs):
        ''' Applies TfidVectorizer fit and transform methods '''
        return self.fit(docs).transform(docs)
    
    def re_fit(self,docs):
        
        ''' Update fitted vectorizer's vocabulary.
            Avoids having to fit new data from scratch.
        
        Parameters
        ----------
        docs: iterable
            contains str to be added
        Returns
        ----------
        self 
        
        '''
        
        docs = self.add_dimensions(docs)
        common,_,c_idx = np.intersect1d(docs,self.docs,False,True)
        new_docs = [d for d in docs if d not in common]
        
        if new_docs: #If we have new documents
            formatter = self.build_analyzer()
            max_vocab = len(self.vocabulary_)
            num_doc = len(self.docs)
            doc_freq = self.doc_freq
            
            tokens = list(chain(*[formatter(item) for token in new_docs for item in token]))
            new_tokens = set(tokens).difference(set(self.vocabulary_))
            
            if new_tokens: #If we have new tokens
                new_tokens_idx = range(max_vocab,max_vocab + len(new_tokens))
                self.vocabulary_.update(dict(zip(new_tokens,new_tokens_idx)))
                doc_freq = np.append(self.doc_freq,[0]*len(new_tokens))
            
            for t in tokens:
                doc_freq[self.vocabulary_[t]] += 1
            
            if self.use_idf == True: #Equivalent to CountVectorizer with normalization
                num_doc = num_doc + len(new_docs)
                idf = np.log(float(num_doc + self.smooth_idf)/(doc_freq + 1)) + self.smooth_idf
                self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))
            
            self.new_docs = np.append(self.new_docs,new_docs)
            self.docs = np.append(self.docs,new_docs)
            self.doc_freq = doc_freq
             
        return self
    
    def re_fit_transform(self,docs):
        ''' Updates fitted vectorizer's vocabulbary and transforms 
        
        Parameters
        ----------
        docs: iterable
            contains str to be added
        Returns
        ----------
        X: Document-term matrix, [n_samples, n_features]
        
        '''
        self.re_fit(docs)
        return self.transform(docs)
    
    def cosine_sim(self,row_idx=slice(None),col_idx=slice(None)):
        ''' Calculate cosine-similarity for document term matrix 
        
        Parameters
        ----------
        row_idx: array-like
        col_idx: array-like
            contains index values for slicing, if none are given
            the equivalent of [:] is passed
        
        Returns
        ----------
        CS: cosine-similarity matrix, [row_idx,col_idx]
        '''
        
        mtx = self.transform(self.docs)
        CS = np.dot(mtx[row_idx],mtx[col_idx].T)
        return CS
    

    def group_string(self,target_strings,min_score =0.1,create_dict=False,restrict_shown=None):
        ''' Output similarity scores for a given set of strings.
            Comparisons are made on the original set of documents
            
        
        Parameters
        ----------
        target_strings: array-like
            contains str 
            
        min_score: array-like
            contains minimum similarity scores for filtering
        
        create_dict: boolean
            if set to True, creates a dictionary between target_strings and 
            similarity output
        
        Returns
        ----------
        full_output: Pandas Dataframe or Dictionary depending on create_dict
                     containing similarity output
                    
        '''

        target_strings = self.add_dimensions(target_strings)
        min_score = self.add_dimensions(min_score)
    
        self.re_fit(target_strings) 
        
        #work out row_idx and col_idx

        c_idx = [np.where(t==self.docs)[0][0] for t in target_strings]
        ex_idx = [np.where(n ==self.docs)[0][0] for n in self.new_docs]
        inc_idx =np.where(~np.in1d(np.arange(len(self.docs)),ex_idx))[0]
        
        sim_mtx = self.cosine_sim(c_idx,inc_idx)
 
        
        #adjust minimum scores to match the size of target_strings
        if min_score.size != target_strings.size:
            ext = [min(min_score)]*(len(target_strings)-len(min_score))
            min_score = np.append(min_score,ext)
        
        
        full_output = pd.DataFrame(columns = ['index','matches','scores'])
        for n,i in enumerate(target_strings):
        
            r_idx = np.where(sim_mtx[n,:].A>min_score[n])[1]
            
            value = ([n]*len(r_idx),
                     self.docs[r_idx],
                     sim_mtx[n,r_idx].A[0])
            
            summary = pd.DataFrame(np.column_stack(value),
                                   columns = full_output.columns)\
                                   .sort_values(by=['scores'],ascending = False)\
                                   .astype({'scores':np.float64})
                        
            full_output = full_output.append(summary.iloc[:restrict_shown,:]).reset_index(drop=True)
        
        if create_dict == True:
            d_items = {row['matches']:target_strings[row['index']] 
                        for _,row in full_output.iterrows()}
            return d_items
        else:
            return full_output
        
        
