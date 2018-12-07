import numpy as np
import pandas as pd

##########################
####data preprocessing####
##########################

def centralize(arra):
    cen = arra - np.mean(arra)
    var = np.sqrt(sum(cen**2)/(len(arra)-1))
    arra = cen/var
    return arra

#read the data
ds0 = pd.DataFrame(pd.read_csv('insurance.csv'))

#check if there is any null
ds0.isnull().sum()
#no null :)

# map the character to factor variable
ds0.sex = ds0.sex.map({'female':1, 'male':0})
ds0.smoker = ds0.smoker.map({'yes':1, 'no':0})
ds_reg = pd.get_dummies(ds0['region']).iloc[:,0:4]
# since we change it to dummy variables, we have to drop one of the column
ds0 = ds0.join(ds_reg.iloc[:,0:3])

# actually centralize is not a must, which would not change the general result of the regression.
# ds0.bmi = centralize(ds0.bmi)
# ds0.age = centralize(ds0.age)

# Take log for the charges because of its heavey tail.
ds0['logcharges'] = np.log(ds0.charges)

# call the prepreocessed data ds1
ds1 = ds0.drop(['charges','region'],axis=1)

####visualization####
import seaborn as sns
import matplotlib.pyplot as plt

# Data in every column looks fine right now.
f,ax = plt.subplots(2,3,figsize=(10,8))
sns.distplot(ds1["age"], kde=False, ax=ax[0,0])
sns.boxplot(x='sex',y='charges', data=ds0, ax=ax[0,1])
sns.distplot(ds1['logcharges'], ax=ax[0,2], kde=False, color='b')
# The logcharges are now normally distributed.
sns.distplot(ds1['bmi'],ax=ax[1,0], kde=False, color='b')
sns.countplot('children',data=ds1, ax=ax[1,1])
sns.countplot('region',data=ds0, ax=ax[1,2])

ax[0,0].set_title('Distribution of Ages')
ax[0,1].set_title('Charges boxplot by Sex')
ax[0,2].set_title('Distribution of log charges')
ax[1,0].set_title('Distribution of bmi')
ax[1,1].set_title('Distribution of children')
ax[1,2].set_title('Distribution of regions')

#########################
####model dignositics####
#########################

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

y = ds1.logcharges
X = ds1.drop(['logcharges'], axis = 1)


# VIF
# The vif of each column is ok. All of them are smaller than 5, even 2.
def variance_inflation_factor(exog, exog_idx):
    k_vars = exog.shape[1]
    x_i = exog.iloc[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog.iloc[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif

# VIF of each column

# we skip the constant column
VIF = [variance_inflation_factor(add_constant(X), i) for i in range(1,X.shape[1]+1)]


regr_1 = OLS(y, add_constant(X)).fit()

# residual distribution
sns.distplot(regr_1.resid) # acting like normal which is good

# since the residual itself is normal, box-cox is not necessary.
# namda = 0.1
# regr_test = OLS((y**namda-1)/namda, add_constant(X)).fit()
# sns.jointplot((y**namda-1)/namda, regr_test.resid)
# sns.distplot(regr_test.resid)

sns.jointplot(y, regr_1.resid) # which looks very strange. maybe the model is not linear at the first place.
#since there is explicit non-linear in this model, we have to add some non-linear covariates in it.

# partial residual plot
# Which attempts to show how covariate is related to dependent variable
# if we control for the effects of all other covariates
# partial residual plots look acceptable.
sns.jointplot(regr_1.params.bmi * X.bmi + regr_1.resid, X.bmi)
sns.jointplot(regr_1.params.age * X.age + regr_1.resid, X.age)


#######################
####model selection####
#######################

#original model
regr_1.summary()

#the first issue we need to solve is residual's dependent problem.
#try NO1: add an interactive covariate smoker:bmi
X_2 = X.iloc[:,:]
X_2['sm_bm'] = X_2.smoker * X_2.bmi
regr_test = OLS(y, add_constant(X_2)).fit()
regr_test.summary()
# which certainly improve the performance of the model

#try NO2: add an interactive covariate smoker:age
X_2['sm_ag'] = X_2.smoker*X_2.age
regr_test = OLS(y, add_constant(X_2)).fit()
regr_test.summary()
# which increase the performance of the model significantly

# since we only have two continuous covariates, we can try to give them an extra power.
#try NO3: add bmi^1.5
X_2['bmi^1.5'] = X_2.bmi ** 1.5
regr_test = OLS(y, add_constant(X_2)).fit()
regr_test.summary()

#try NO4: add age^1.5
X_2['age^1.5'] = X_2.age ** 1.5
regr_test = OLS(y, add_constant(X_2)).fit()
regr_test.summary()

#try NO5: what if we cancel the log?
y_2 = ds0.charges
regr_test = OLS(y_2, add_constant(X_2)).fit()
regr_test.summary()

#Now we have added all the possible covariates we can, we can consider drop some of them.

#As we can see from the summary, it seems that the region para is not that important.

X_3 = X_2.drop(columns = ['northwest', 'southeast'])
regr_3 = OLS(y_2, add_constant(X_3)).fit()
regr_3.summary()
# the AIC and BIC actually get smaller.

#similarly smoker:age is not that important either.
X_4 = X_3.drop(columns = ['sm_ag'])
regr_4 = OLS(y_2, add_constant(X_4)).fit()
regr_4.summary()

#after the model selection part, we now have our residual:
sns.residplot(np.sum(regr_4.params*X_4,1)+regr_4.params[0], y_2)
#which is much more acceptable than before.

# For now, the best model would be
# charges = constant + age + sex + bmi + children + smokeryes + northeastornot + sm_bm + bmi^1.5 + age^1.5
# caution: as you may see, the coefficient of smokeryes is negative.
# Please don't take the result as smoking is good for your health. Since smoke is also used in smoke*bmi, bmi is large.
# So the bad influence of smoking has been transferred to the smoke*bmi term.

# const        -7441.291486
# age           -323.193018
# sex            509.876229
# bmi           1075.107329
# children       681.686500
# smoker      -20425.253300
# northeast     1036.802773
# sm_bm         1443.949117
# bmi^1.5       -126.749809
# age^1.5         62.573541
# R-squared       0.844