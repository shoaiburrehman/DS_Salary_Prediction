# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 04:07:38 2020

@author: S. Khan
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv('../eda_sal.csv')
df
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','numof_comp','hourly','employer_provided',
             'job_state','same_state','age','python_job','spark_job','aws','excel','job_simp','seniority','desc_len']]

df_dum = pd.get_dummies(df_model)
df_dum

from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)

lm_l_predict = lm_l.predict(X_test)

#print(f"Train accuracy {round(lm_l.score(X_train,y_train)*100,2)} %")
#print(f"Test accuracy {round(lm_l.score(X_test,y_test)*100,2)} %")

np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)


err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))


# models GridsearchCV 
#from sklearn.model_selection import GridSearchCV
#parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

#gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
#gs.fit(X_train,y_train)


#gs.best_score_

#gs.best_estimator_

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
#tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tpred_lm) #mae for linear regression

mean_absolute_error(y_test,tpred_lml) #mae for Lasso regression

#mean_absolute_error(y_test,tpred_rf) #mae for random forest regression

#mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

#accuracy for linear regression
#print(f"Train accuracy {round(lm.score(X_train,y_train)*100,2)} %")
#print(f"Test accuracy {round(lm.score(X_test,y_test)*100,2)} %")

#accuracy for lasso regression
#print(f"Train accuracy {round(lm_l.score(X_train,y_train)*100,2)} %")
#print(f"Test accuracy {round(lm_l.score(X_test,y_test)*100,2)} %")

#accuracy for random forest regression
#print(f"Train accuracy {round(gs.best_estimator_.score(X_train,y_train)*100,2)} %")
#print(f"Test accuracy {round(gs.best_estimator_.score(X_test,y_test)*100,2)} %")

#import pickle
#pickl = {'model': gs.best_estimator_}
#pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

#file_name = "model_file.p"
#with open(file_name, 'rb') as pickled:
#    data = pickle.load(pickled)
#    model = data['model']
    
#X_test.index.iloc

#model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

def predic_index(x):
    return list(X_test.iloc[x,:])

def salary_estimate(x):
    for i in df.index.values:
        if X_test.index[x] == df.index.values[i]:
            rating = df['Rating'][i]
            salary_min = df['min_salary'][i]
            salary_max = df['max_salary'][i]
            salary_avg = df['avg_salary'][i]
            job_title = df['Job Title'][i]
            sal_est = df["Salary Estimate"][i]
            break
    return sal_est

predic_index(1)


#we just testing
