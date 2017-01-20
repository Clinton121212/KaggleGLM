# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:43:09 2016

@author: clinton1212
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing


class big_skl_chunk(object):
    '''
    wrapper around sklearn objects (already initialized)
    takes chunked pandas object as training
    predicts dataframe
    
    
    '''
    def __init__(self,CLFs,response,use_vars=None):
        '''
        Args:
           CLFs (sklearn object): list of sklearn clasifiers (initialized)
           response (str): string denoting Y variable (columns heading)
           use_vars(list): list of columns string names  to use in X
        
        '''
        self.list_clf=CLFs
        self.response=response
        self.use_vars=use_vars

    def fit(self,chunks):
        '''
        predict general
        assumes chunk in chunks for each classifier
    
        Args:
            chunks(pandas): pandas chunked object pre formatted
        '''
        f=0
        
        for chunk in chunks:
            if self.use_vars is None:
                self.use_vars=set(chunk.columns).remove(self.response)   
            X=chunk[self.use_vars]
            Y=chunk[self.response]
        
            self.list_clf[f].fit(X,Y)
            print(f)
            f=f+1
        return self
        
    def predict(self,X):
        X=X[self.use_vars]
        if True:
            norm_fact=len(self.list_clf)
            first=True
            for clf in self.list_clf:
                if first:
                    pred=clf.predict(X)/norm_fact
                    first=False
                else:
                    
                    pred=pred+clf.predict(X)/norm_fact
                
            return pred
        
            


def log_metric(Y,pred):
    '''
    
    returns sum  (log (1+pred)-log(1+actual))^2
    
    
    Y: dataseries
    pred: array or data sereis
    
    returns: score (value)
    '''
    score=np.mean((np.log(1+pred)-np.log(1+Y))**2)
    return score

def send_to_gropo_sub(CLF,Test,use_vars):
    '''
    Send grupo data to submission (takes fitted object give csv for submission)
    
    Test=chunked data frame
    CLF:classifier trained
    use_vars: columns to use in test in prediction
    '''
    first=True
    
    for test in Test:
       
        
        hold_id=test['id']
        del test['id']
        test.columns=  ['a','b','c','d','e','f']     
        testt=test[use_vars]
        pred=CLF.predict(testt)
        pred=pd.DataFrame(pred)
        
        
        if first:    
            df=pd.DataFrame({'id':hold_id})
            pred.index=df.index
            df['Demanda_uni_equil']=pred
            first=False
        else:
            print(first)
            df2=pd.DataFrame({'id':hold_id})
            pred.index=df2.index
            df2['Demanda_uni_equil']=pred
        
            df=df.append(df2)

    
    df.to_csv('sub.csv',index=False)


def junk_test_SMWrapepr(PATH):
    '''
    Junk to test wrapper for stats model glm
    '''
    
    df=pd.read_csv(PATH,nrows=100,\
    usecols=['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',\
                            'Producto_ID','Demanda_uni_equil'])
                            
    family_link = sm.families.Poisson(sm.families.links.log)
    new_names=['a','b','c','d','e','f','g']
    df.columns=new_names  
    
    
    sd=['a','b','c','d','e','f' ]    
    formula='Y ~ ' 
    tru=True
    for j in sd:
        if tru:
            tru=False
            formula = formula +  j
        else:
            formula=formula +  ' + ' +  j
    clf=SMWrapper_ci(smf.glm,formula=formula,family=family_link)    

class SMWrapper_ci(object):
    '''
    Class wrapper around stats model object to make it behave like sklearn object
    Must initializ  statsmodels with formula 'Y~...' (must be Y for response)
    '''
    def __init__(self,mdl,**arg):
        '''
        Initializes statsmodel object so that it may be 
        fit / predicted with only X,Y data as
        is the case with sklearn.
        
        
        mdl : uninitialized stats model object
        args : arguments to be based to stats_model object 
        '''
        self.mdl=mdl
        self.args=arg
        
    def predict(self,X):
        '''
        Calls predict with classifier
        '''
        return self.clf.predict(X)

    def predict_proba(self,X):
        return self.clf.predict(X)
        
    def fit(self,X,Y):
        Y.index=X.index
        X['Y']=Y

        self.clf=self.mdl(data=X,**self.args,).fit() # maybe **args...
class Map_mean_fit_wrapper(object):
    '''
    wrapper to fit around initialized sklearn object.
    Maps variables to mean response 
    Useful for multi-categorical variables (like person id an dpostal code)
    '''
    def __init__(self,mdl, vars_to_map, mapping_version):
        '''
        initializes sklearn object
        
        mdl: initialized sklearn object
        maping_version: string denoting how to map data ####to be developed 
        just call 'mapping versionTrue'
        
        vars_to_map: list of column headings to map to mean
        '''
        self.maps=mapping_version
        self.clf=mdl
        self.map_cols=vars_to_map
        
    def fit(self,X,Y):
        '''
        calls sklearn formatted fit after mapping data
        X: pandas dataframe X variables
        Y: pandas dataframe with response
        '''
        X_hold=X[:]
        
        X_mapped, Y =self.data_map1(X[self.map_cols][:],Y[:])
        X_mapped.index=X.index
        X_hold[self.map_cols]=X_mapped[:]
        self.maxes=np.max(X_hold,axis=0) #for capping data
        self.clf.fit(X_hold[:],Y[:])
    def predict(self,X):
        '''
        calls sklearn formatted predict after mapping data
        X: pandas dataframe
        '''
        
        
        X_hold=X[:]
        
        X_mapped=self.data_map2(X[self.map_cols][:])
        X_mapped.index=X.index        
        X_hold[self.map_cols]=X_mapped[:]
        
        #############################fix to vecctorize##############
        keeep=X_hold>self.maxes
        for cols in X_hold.columns:
            X_hold[keeep[cols]]=self.maxes.ix[cols]   
            
        #############################################################    
        
        pred=self.clf.predict(X_hold[:])
        return pred
        
    def data_map1(self,X,Y):
        '''
        used in self.fit()
        must be called prior to  data_map2
        
        default maps X values to mean Y value for variable
        saves #To be developed
        
        X: pandas dataframe
        Y: pandas dataframe
        '''
        collumns=X.columns
        X['Y']=Y

        sb=X[:]
        self.mappings=[]
    
        for cols in collumns:
            gb=sb.groupby(cols,sort=False)
            dats=gb.mean()
            dats=dats.reset_index()
            dummy_col=list(dats.columns)
            dummy_col.remove('Y')
            dummy_col.append('y')
            dats.columns=dummy_col
            self.mappings.append(dats)
            X['Order'] = np.arange(len(X))

            X = X.merge(dats[[cols,'y']], how='left', on=[cols])\
            .set_index("Order").ix[np.arange(len(X)), :]
            
            #maps dats into X,
            X[cols]=X['y']
            del X['y']
        Y=X['Y']
        del X['Y']
        
        if True:
            #normalizes
            self.scaler=preprocessing.StandardScaler().fit(X)
            X=self.scaler.transform(X[:])
            X=pd.DataFrame(X)
            X.columns=collumns
        return X[collumns], Y[:]
        
    def data_map2(self,X):
        '''
        used in self.predict()
        must be called after data_map1
        
        X: pandas dataframe with same column names as was used in fitting
        
        '''
        sb=X[:]
        collumns=X.columns #assumes same order of columns
        countt=0
        for cols in collumns:
            dats=self.mappings[countt]
            
            X['Order'] = np.arange(len(X))

            X = X.merge(dats[[cols,'y']], how='left', on=[cols])\
            .set_index("Order").ix[np.arange(len(X)), :]
           
            X[cols]=X['y']
            del X['y']    
            countt=countt+1
            
            
        ############################ Fixed...but bad..vectorize###################
        if False:
            #modify later with more options
            means=X.mean(axis=1)            
            means=means.fillna(means.mean(axis=0))
            means.index=X.index
            for i in X.columns:
                hold=X[i]
                hold[hold.isnull()]=means[hold.isnull()]
                X[i]=hold
        if True:
            #modify later with more options
            means=X.mean(axis=1)            
            means=means.fillna(means.mean(axis=0))
            means.index=X.index
            for i in X.columns:
                hold=X[i]
                hold[hold.isnull()]=means[hold.isnull()]
                X[i]=hold
                
        X=self.scaler.transform(X)
        X=pd.DataFrame(X)
        X.columns=collumns
        return X        
##############################################################        

