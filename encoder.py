# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:46:34 2019
@author: Samuel
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pickle
import six
from itertools import cycle
from collections import OrderedDict
from scipy.sparse import csr_matrix
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import warnings


class DiscreteMinxin(object):
    def get_obj_cols(self, df):
        dtypes_to_encode=['object','category']
        cols = df.select_dtypes(include=dtypes_to_encode).columns.tolist()
        return cols
    
    def save(self, path):
        with open(path,'wb+') as fp:
            pickle.dump(self.__dict__, fp)
            
    def load(self,path):
        with open(path,'rb') as fp:
            dict_param = pickle.load(fp)
        return dict_param
            
class OneHotEncoder(BaseEstimator, TransformerMixin, DiscreteMinxin):
    
    def __init__(self, dummy_na=True, handle_unknown='ignore',
                 category_threshold=50, drop_threshold_cols=True,
                 replace_na='null'):
        """
        parameter
        ---------
        dummy_na: bool, defualt True
        handle_unknown: str, 'error' or 'ignore'
        category_threshold: columns of categories more then this threhold will
                            not be encoded
        drop_threshold_cols: drop columns that not satisfy category_threshold 
                             or columns of one category
        """
        self.dummy_na = dummy_na
        self.handle_unknown = handle_unknown
        self.category_threshold = category_threshold
        self.drop_threshold_cols = drop_threshold_cols
        self.encode_cols= []
        self.drop_cols=[]
        self.mapping = {}
        self.replace_na = replace_na
        self._dim = None
        
    def fit(self, X, y=None, cols_to_encoder=None, extra_numeric_cols=None):
        """
        parameter
        ----------
        X: DataFrame obj to generate one-hot-encoder rule
        
        cols_to_encoders: specify the columns  to be  encoded
        extra_numeric_cols: if cols_to_encoder is provided this param will
           not be used, otherwise all object columns and extra_numeric_cols 
           will be encoded.
        """
        if not isinstance(X, DataFrame): 
            raise TypeError('X should be DataFrame object')
        
        if y is not None:
            if y not in X.columns:
                raise ValueError('y is not in X.columns during fit')
            self._dim = X.shape[1] -1
        else:
            self._dim = X.shape[1]
            
        # get encoder columns 
        if cols_to_encoder is None:
            cols = self.get_obj_cols(X)
            if extra_numeric_cols is not None:
                cols += list(extra_numeric_cols)
        else:
            cols = cols_to_encoder   
            
        cols = list(set(cols))
        if len(cols)==0:
            print('no colums to encoder')
            return
        
        if y in cols:
            cols.remove(y)
            
        # re-order cols by original order 
        cols = sorted(cols, key=X.columns.get_loc)       
        # convert na to nullsym for sake of simplicity
        df = X[cols].fillna(self.replace_na)
        
        # generato rules 
        cats_list = pd.Series()
        for col in cols:
            cats_list[col] = df[col].unique().tolist()
            if (not self.dummy_na) and (self.replace_na in cats_list[col]):
                cats_list[col].remove(self.replace_na) 
        print(cats_list)      
        cats_cnt = cats_list.apply(lambda x: len(x))
        # exclude columns of too manay categories or just one category
        drop_mask = (cats_cnt > self.category_threshold) | (cats_cnt==1)
        drop_index = cats_cnt[drop_mask].index
        cats_list = cats_list[~cats_list.index.isin(drop_index)]
#        if self.sort_value:
#            cats_list = cats_list.apply(lambda x: sorted(x))
        
        self.drop_cols = drop_index.tolist()
        self.encode_cols = cats_list.index.tolist()
        
        maps={}
        for col in self.encode_cols:
            # map each val in col into a index
            vallist = cats_list[col]   
            validx = OrderedDict({str(val):i for i,val in enumerate(vallist)})
            maps[col] = validx
            
        self.mapping = maps
        
        # convert back to na 
        #X[cols].replace({self.replace_na:np.nan},inplace=True)
        
        
    def transform(self, X, y=None, dtype=None, inplace=False): 
        """
        parameter
        -----------
        dtype: specifies the dtype of encoded value
        """
        
        if not isinstance(X, DataFrame):
            raise TypeError('X shoule be DataFrame object')
        if y is not None:
            if y not in X.columns:
                raise ValueError('y not in X.column during transform')
            if self._dim != X.shape[1] -1:
                raise ValueError('dimension error')
        elif self._dim != X.shape[1] :
            raise ValueError('dimension error')
        
        if not inplace:
            X = X.copy() # X=X.copy(deep=True)
            
        if self.drop_threshold_cols:
            X.drop(self.drop_cols,axis=1, inplace=True)
            
        data_to_encode = X[self.encode_cols].fillna(self.replace_na)
        with_dummies = [X.drop(self.encode_cols,axis=1)]
        
        prefix = self.encode_cols
        prefix_sep = cycle(['_'])
        
        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (col_name, col_series) type
            dummy = self._encode_column(col[1], pre, sep, dtype = dtype)
            with_dummies.append(dummy)
            
        result = pd.concat(with_dummies, axis=1)
        
        return result
      
    def _encode_column(self, data, prefix, prefix_sep, dtype):
        
        if dtype is None:
            dtype = np.uint8
            
        maps = self.mapping[prefix]
        dummy_strs = cycle([u'{prefix}{sep}{val}'])
        dummy_cols = [dummy_str.format(prefix=prefix,sep=prefix_sep,val=str(v))
                      for dummy_str, v in zip(dummy_strs, maps.keys())]
        
        if isinstance(data, Series):
            index = data.index
        else:
            index = None
            
        row_idxs= []
        col_idxs= []
        for i, v in enumerate(data):
            idx = maps.get(str(v),None)
            if idx is None:
                print("{} only exist in test column '{}'".format(v, prefix))
            else:
                row_idxs.append(i)
                col_idxs.append(idx)
        sarr = csr_matrix((np.ones(len(row_idxs)),(row_idxs,col_idxs)),shape=
                          (len(data),len(dummy_cols)), dtype=dtype)
        
        out = pd.SparseDataFrame(sarr, index=index, columns=dummy_cols,
                           default_fill_value=0,dtype=dtype)
        
        return out
        
    
    
class MeanEncoder(BaseEstimator, TransformerMixin, DiscreteMinxin):
    def __init__(self, dummy_na = True, handle_unknown='prior', n_critical=1, 
                 scale_factor=1, drop_last = False, replace_na= -99):
        """
        dummy_na: bool,if False the null values will be repaced with prior after
                  transform
        handle_unknown: str, 'error' of 'prior'
        drop_last: bool,whether to get C-1 categories out of C by removing the
                   last class.
        n_critical: the critical point that the posterior will contribute more
        scale_factor: scale the smoothing factor
        replace_na : int
        """
        self.dummy_na = dummy_na
        self.handle_unknown =handle_unknown
        self.n_critical = n_critical
        self.scale_factor = scale_factor
        self.drop_last = drop_last
        self.mapping={}
        self.prior=None
        self.encode_cols= None
        self.replace_na = replace_na  # 
        self._dim =None     # attribution dimension
        
    def fit(self, X, y, cols_to_encode=None, extra_numeric_cols=None):
        
        if not isinstance(X, DataFrame):
            raise ValueError('X should be DataFrame type')
        
        if isinstance(y, six.string_types):
            if y in X.columns:
                self._dim = X.shape[1] - 1
                X = X.rename(columns={y:'_y_'})
            else:
                raise ValueError('y not in X.columns during fit')
        else:
            self._dim = X.shape[1]
            y = pd.Series(y, name='_y_')
            X = X.join(y)
            
        X['_y_'] = X['_y_'].astype(int)   
        
        # get encoder columns 
        if cols_to_encode is None:
            cols = self.get_obj_cols(X)
            if extra_numeric_cols is not None:
                cols += list(extra_numeric_cols)
        else:
            cols = cols_to_encode    
        cols = list(set(cols))
        
        if len(cols)==0:
            print('no colums to encoder')
            return     

        # re-order cols by original order 
        cols = sorted(cols, key=X.columns.get_loc)
        self.encode_cols = cols
        
        data_to_encode = X[self.encode_cols+['_y_']]
        # convert na to a predefined value 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
        prior = data_to_encode['_y_'].value_counts()/len(data_to_encode['_y_'])
        prior.sort_index(axis=0,inplace=True)
        prior.name='prior'
        self.prior = prior  #series
        
        maps = {}
        for col in self.encode_cols:
            ctb = pd.crosstab(index=data_to_encode[col], columns=data_to_encode['_y_'])
            # deal with missing y.
            ctb = ctb.reindex(columns=prior.index, fill_value = 0)
            ctb.sort_index(axis=1,inplace=True)
            # calculate posterior 
            post = ctb.apply(lambda x: x/x.sum(), axis =1)
            # calcalate smoothing factor of prior and posterior
            smooth = ctb.applymap(lambda x: 1/(1+np.exp(-(x-self.n_critical)/self.scale_factor)))
            smooth_prior = (1-smooth).multiply(prior,axis=1) # DataFrame multiple series
            smooth_post =  smooth.multiply(post)
            codes = smooth_prior + smooth_post
            # normalize
            codes = codes.divide(codes.sum(axis=1),axis=0)
            # encode na with prior if na is not treated as a cateogry
            if not self.dummy_na and self.replace_na in codes.index:
                codes.loc[self.replace_na,:]=self.prior
            maps[col] =codes
            
        self.mapping = maps
            
    def transform(self, X, y=None):
        if not isinstance(X, DataFrame):
            raise ValueError('X should be DataFrame type')
        if isinstance(y, six.string_types) and y in X.columns:
            if  self._dim != X.shape[1] -1:
                raise ValueError('dimension error')
        elif self._dim != X.shape[1]:
            raise ValueError('dimension error')
        
        if not self.encode_cols:
            return X
        
        data_to_encode = X[self.encode_cols]
        #fill na 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
            
        with_dummies = [X.drop(self.encode_cols,axis=1)]
        
        prefix = self.encode_cols
        prefix_sep = cycle(['_'])
        
        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (col_name, col_series) type
            dummy = self._encode_column(col[1], pre, sep)
            with_dummies.append(dummy)
            
        result = pd.concat(with_dummies, axis=1)
        
        return result
    
    def _encode_column(self, data, prefix, prefix_sep):
              
        maps = self.mapping[prefix]
        dummy_strs = cycle([u'{prefix}{sep}{val}'])
        dummy_cols = [dummy_str.format(prefix=prefix,sep=prefix_sep,val=str(v))
                      for dummy_str, v in zip(dummy_strs, maps.columns)]
        
        if isinstance(data, Series):
            index = data.index
        else:
            index = None
        
        enc_df = maps.loc[data.values,:] # NaN with unknonw value
        #handle unknown value
        if not all(data.isin(maps.index)):
            msg = "unknown category {} in column '{}'".format(
                        data[~data.isin(maps.index)].values, prefix)
            if self.handle_unknown=='error' :
                raise ValueError(msg)
            else:
                print(msg)
                enc_df.fillna(self.prior,inplace=True)  
                
        enc_df.index = index
        enc_df.columns = dummy_cols
        if self.drop_last:
            enc_df = enc_df.iloc[:,:-1]
            
        return enc_df
        
    def _fill_na(self, df, cols):
        
        dtypes = df.dtypes
        for col in cols:
            if any(df[col].isna()) :
                df.loc[:, col] = df[col].fillna(self.replace_na)
                if dtypes[col].name.startswith('float'):
                    df.loc[:,col] = df[col].astype(int)
                elif dtypes[col].name == 'object':
                    df.loc[:,col] = df[col].astype(str)

class WoeEncoder(BaseEstimator, TransformerMixin, DiscreteMinxin):
    """
     currently only support discrete variable encode.
    """
    def __init__(self, dummy_na = True, handle_unknown='zero', replace_na=-99,
                 reg = 1):
        '''
        dummy_na bool, force to True
        handle_unknown: one of ('zero', 'error')
        reg: int, bayesian prior value  to avoid divding by zero when calculate woe.
        
        '''
        self.dummy_na = dummy_na
        self.handle_unknown = handle_unknown
        self.replace_na = replace_na
        self.mapping ={}
        self.reg = reg
        self._dim = None
        
    def fit(self, X, y, cols_to_encode = None, extra_numeric_cols=None):
        
        if not isinstance(X, DataFrame):
            raise ValueError('X should be DataFrame type')
        
        if isinstance(y, six.string_types):
            if y in X.columns:
                self._dim = X.shape[1] - 1
                X = X.rename(columns={y:'_y_'})
            else:
                raise ValueError('y not in X.columns during fit')
        else:
            self._dim = X.shape[1]
            y = pd.Series(y, name='_y_')
            X = X.join(y)
        # target label as '_y_'
        X['_y_'] = X['_y_'].astype(int)   
        
        # get encoder columns 
        if cols_to_encode is None:
            cols = self.get_obj_cols(X)
            if extra_numeric_cols is not None:
                cols += list(extra_numeric_cols)
        else:
            cols = cols_to_encode    
        # re-order cols by original order 
        self.encode_cols = sorted(list(set(cols)), key=X.columns.get_loc)
        
        if len(self.encode_cols)==0:
            print('no colums to encoder')
            return     

        data_to_encode = X[self.encode_cols+['_y_']]
        # convert na to a predefined value 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
        
        self._pos = data_to_encode['_y_'].sum()  # global positive count
        self._neg = len(data_to_encode['_y_']) - self._pos # global negative count 
        
        maps ={}
        for col in self.encode_cols:
            woe = self._compute_woe(data_to_encode, col, '_y_') # return series
            maps[col] = woe
            
        self.mapping = maps
            
    def _compute_woe(self, df, var, y='_y_'):
        grp = df[y].groupby(df[var]).agg(['sum',lambda x: x.count()-x.sum()])
        grp = grp.rename(columns={'sum':'pos', '<lambda>':'neg'})
        
        #bayesian prior to avoid divide by zero
        woe = np.log((grp['pos']+self.reg)/(grp['neg']+self.reg)) - \
              np.log((self._pos+2*self.reg)/(self._neg+2*self.reg))
        return woe
    
    def transform(self, X, y=None, inplace=False):
        if not isinstance(X, DataFrame):
            raise ValueError('X should be DataFrame type')
        if isinstance(y, six.string_types) and y in X.columns:
            if  self._dim != X.shape[1] -1:
                raise ValueError('dimension error')
        elif self._dim != X.shape[1]:
            raise ValueError('dimension error')
        
        if not self.encode_cols:
            return X
        
        if not inplace:
            X = X.copy()
        
        X[self.encode_cols] = X[self.encode_cols].fillna(self.replace_na,
                                     downcast = 'infer')
        print(X)
        msg = "unseen category {} in column '{}'"
        for col in self.encode_cols:
            X[col] = X[col].map(self.mapping[col])
            #handle unknown value
            if any(X[col].isnull()):
                if self.handle_unknown == 'error':
                    raise ValueError(msg.format(X[X[col].isnull()][col].values, col))
                else:
                    print(msg.format(X[X[col].isnull()][col].values, col))
                    X[col] = X[col].fillna(0.0)                   
        return X     
               
if __name__== '__main__':
    warnings.filterwarnings('ignore')
    df = pd.DataFrame([[1,2,3,'a',2.0],[1,np.nan,4,'6',3.0],[2,3,4,5,6],
                       [2.0,3,4,5,np.nan]],columns=['x','y','z','j','k'])
    df.index=['1','2','3','4']
    #df = pd.DataFrame([[1,2,3],[3,4,5]])
#    ohe = OneHotEncoder(dummy_na=False)
#    ohe.fit(df,extra_numeric_cols=['y','x','j'])
#    ret = ohe.transform(df)
#    print(df)
#    print(ret)
    df = pd.DataFrame([[1,2,3],[np.nan,3,4],[2,3,6],[2,4,2]],columns=['a','b','c'])
    y = pd.Series([1,0,0,1])
    
    mec = MeanEncoder(drop_last=False,dummy_na=True)
    mec.fit(df,y,extra_numeric_cols=['a','b'])
    print(mec.transform(df))
    df2 = pd.DataFrame([[1,2,3],[np.nan,3,4],[3,3,6],[2,4,2]],columns=['a','b','c'])
    print(mec.transform(df2))
    
    print('woe encdoer'.center(40,'-'))
    wec = WoeEncoder()
    wec.fit(df,y, extra_numeric_cols=['a','b','c'])
    print(wec.transform(df))
