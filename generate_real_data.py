#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


# In[9]:


def generate_data(n, ml, scale = 1.0):
    n_mv = n
    n_us = n
    mv_s = ml['movieId'].value_counts()[0:n_mv]
    us_s = ml['userId'].value_counts()[0:n_us]
    mv_df = pd.DataFrame(columns = ['movieId','mvId'])
    mv_df['movieId'] = mv_s.index
    mv_df['mvId'] = np.arange(n_mv)
    us_df = pd.DataFrame(columns = ['userId','usId'])
    us_df['userId'] = us_s.index
    us_df['usId'] = np.arange(n_us)
    mg1 = pd.merge(left = mv_df,
        right = ml,
        on = 'movieId',
        how = 'left')
    mg2 = pd.merge(left = us_df,
        right = mg1,
        on = 'userId',
        how = 'left')
    
    #missing_replacement = np.mean(np.log(mg2['rating']))
    overall_mean = np.mean(mg2['rating'])
    
    mg2['rating'] = mg2['rating'] - overall_mean
    
    mat = np.zeros((n_mv, n_us), dtype=float)
#    mat[mg2['mvId'].values, mg2['usId'].values] = np.log(mg2['rating'].values)
    mat[mg2['mvId'].values, mg2['usId'].values] = mg2['rating'].values * scale
    print(f'Matrix has {mg2.shape[0]/(n_mv*n_us)} observed entries')
    return mat


# In[10]:


ml = pd.read_csv('data/rating.csv')


# In[11]:


n = 50
mat = generate_data(n, ml, scale = 1.6)
np.savetxt(f'data/ratings-{n}x{n}.csv', mat, delimiter=",")


# In[12]:


n = 100
mat = generate_data(n, ml, scale = 1.8)
np.savetxt(f'data/ratings-{n}x{n}.csv', mat, delimiter=",")

