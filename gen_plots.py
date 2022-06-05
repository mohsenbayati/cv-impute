#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import seaborn as sns; sns.set(style='white',rc={"figure.dpi":300, 'savefig.dpi':300},)


# ### Helper functions         

# In[4]:


def lighten_color(color, amount=0.5):  
    # --------------------- SOURCE: @IanHincks ---------------------
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def main_rel_error_plot(data,
                        file_name = None,
                        estimators = ['cv', 'oracle', 'theory-1', 'theory-2', 'theory-3']):
    filter = data['Estimator'].apply(lambda x: x in estimators)
    data4plot = data.loc[filter,:].reset_index(drop=True)
    fig, ax = plt.subplots()
    sns.lineplot(x='n_sample', 
             y ='error', 
             data = data4plot, 
             hue = 'Estimator')
    plt.ylim([0,1])
    plt.xlabel(r'$n$')
    plt.ylabel('Relative Error')
    if file_name is not None:
        plt.savefig(f'plots/{file_name}.pdf')
    else: 
        plt.show()  


def main_lambda_plot(data,
                    file_name = None,
                    estimators = ['cv', 'oracle', 'theory-1', 'theory-2', 'theory-3']):
    
    n_est = len(estimators)
    n_n = data['n_sample'].nunique()

    filter = data['Estimator'].apply(lambda x: x in estimators)
    data4plot = data.loc[filter,:].reset_index(drop=True)

    fig, ax = plt.subplots()
    sns.boxplot(x='n_sample', y ='alpha_opt', 
                data = data4plot,  
                hue = 'Estimator',
                showfliers = False,
                #width=0.5,
                saturation = .7,
                #aspect = 1.5
               )
    for i,patch in enumerate(ax.patches):
        face_col = patch.get_facecolor()
        patch.set_facecolor(lighten_color(face_col, .5))
        patch.set_linewidth(.5)
        
    for i,patch in enumerate(ax.get_legend().get_patches()):   
        face_col = patch.get_facecolor()
        col = face_col
        median_col = lighten_color(face_col, 1)    
        for n in range(5):
            for j in range(n*n_est*n_n+i*5,n*n_est*n_n+i*5+5):
                line = ax.lines[j]
                line.set_color(col)
                line.set_linewidth(.5)
                # Treat median separately
                if (j%5==4):
                    line.set_color(median_col)
                    line.set_linewidth(3)       
    plt.xlabel(r'$n$')
    plt.ylabel(r'Selected $\lambda$')
    plt.yscale('log')
    if file_name is not None:
        plt.savefig(f'plots/{file_name}.pdf')
    else: 
        plt.show()
    
    
    
def read_and_prep_data(filename):
    data = pd.read_csv(filename)
    data.rename(columns={'name':'Estimator'},inplace=True)
    mapping = {'cv':'cv',
           'cv-refit':'cv refit',
           'cv-single': 'cv 1 fold',
           'cv-overfit': 'cv overfit',
           'oracle': 'oracle', 
           'theory-1': 'theory-1',
           'theory-2': 'theory-2',
           'theory-3': 'theory-3'           
          }
    data['Estimator'] = data['Estimator'].apply(lambda x: mapping[x])
    return (data)


# ### Make Graphs

# In[6]:


settings = {'run-100-n-fold-10-sd-p5-mnr-100-100-3-step-100-max-size-2500-seed-10-v2',
            'run-100-n-fold-10-sd-1-mnr-100-100-3-step-100-max-size-2500-v2',
            'run-100-n-fold-10-sd-2-mnr-100-100-3-step-100-max-size-2500-v2',
            #-----------------------            
            'run-100-n-fold-10-sd-p5-mnr-50-50-2-step-100-max-size-1000-v2',
            'run-100-n-fold-10-sd-1-mnr-50-50-2-step-100-max-size-1000-v2',
            'run-100-n-fold-10-sd-2-mnr-50-50-2-step-100-max-size-1000-v2',
            #-----------------------            
            'run-real-100-n-sd-p5-fold-10-50-step-100-max-size-1000-centered',
            'run-real-100-n-sd-1-fold-10-50-step-100-max-size-1000-centered',
            'run-real-100-n-sd-2-fold-10-50-step-100-max-size-1000-centered',
            #-----------------------                        
            'run-real-100-n-sd-p5-fold-10-100-step-100-max-size-2500-centered',
            'run-real-100-n-sd-1-fold-10-100-step-100-max-size-2500-centered',
            'run-real-100-n-sd-2-fold-10-100-step-100-max-size-2500-centered',            
           }


for file_name in settings:
    
    data = read_and_prep_data(f'outputs/{file_name}.csv')
    print('working on',file_name)

    if (np.max(data['n_sample'])<2000):
        jump = 200
    else:
        jump = 500
  
    main_rel_error_plot(data, file_name = f'{file_name}_main_rel_err')
    main_lambda_plot(data.loc[data['n_sample'].apply(lambda x: x%jump)==0,:].reset_index(drop=True), 
                     file_name = f'{file_name}_main_lam_sel')
    
    
    main_rel_error_plot(data, file_name = f'{file_name}_full_rel_err',
                       estimators = data['Estimator'].unique())
    est = data['Estimator'].unique()
    main_lambda_plot(data.loc[data['n_sample'].apply(lambda x: x%jump)==0,:].reset_index(drop=True), 
                     file_name = f'{file_name}_full_lam_sel',
                       estimators = est[est!='cv refit'])


# In[ ]:


plt.close('all')

