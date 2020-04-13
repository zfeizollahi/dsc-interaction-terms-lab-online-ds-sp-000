
# Interactions - Lab

## Introduction

In this lab, you'll explore interactions in the Boston Housing dataset.

## Objectives

You will be able to:
- Implement interaction terms in Python using the `sklearn` and `statsmodels` packages 
- Interpret interaction variables in the context of a real-world problem 

## Build a baseline model 

You'll use a couple of built-in functions, which we imported for you below: 


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Import the Boston data set using `load_boston()`. We won't bother to preprocess the data in this lab. If you still want to build a model in the end, you can do that, but this lab will just focus on finding meaningful insights in interactions and how they can improve $R^2$ values.


```python
regression = LinearRegression()
boston = load_boston()
```


```python
#My code
boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
X = boston_features
y = pd.DataFrame(boston.target, columns=['Target'])
all_data = pd.concat([y,X], axis = 1)
```

Create a baseline model which includes all the variables in the Boston housing data set to predict the house prices. Then use 10-fold cross-validation and report the mean $R^2$ value as the baseline $R^2$.


```python
## my code here
cross_validation = KFold(n_splits=10, shuffle=True, random_state=1)
baseline = np.mean(cross_val_score(regression,X,y,scoring='r2', cv=cross_validation))
```


```python
baseline
```




    0.7190106820189477



## See how interactions improve your baseline

Next, create all possible combinations of interactions, loop over them and add them to the baseline model one by one to see how they affect the $R^2$. We'll look at the 3 interactions which have the biggest effect on our $R^2$, so print out the top 3 combinations.

You will create a `for` loop to loop through all the combinations of 2 predictors. You can use `combinations` from itertools to create a list of all the pairwise combinations. To find more info on how this is done, have a look [here](https://docs.python.org/2/library/itertools.html).


```python
from itertools import combinations
combinations = list(combinations(boston.feature_names, 2))
```


```python
#mycode
interaction_combos = {}
for combo in combinations: 
    data = X.copy()
    interaction = data[combo[0]] * data[combo[1]]
    #interaction = boston_features[combinations[0][0]] * boston_features[combinations[0][1]]
    interaction_name = combo[0] + "_"+ combo[1]
    #interaction_name = combinations[0][0] + "_"+ combinations[0][1]
    data['interaction'] = interaction
    cross_validation = KFold(n_splits=10, shuffle=True, random_state=1)
    baseline = np.mean(cross_val_score(regression,data,y,scoring='r2', cv=cross_validation))
    interaction_combos[interaction_name] = baseline
```


```python
## code to find top 3 interactions by R^2 value here
[[k, v] for k, v in sorted(interaction_combos.items(), key=lambda item: item[1], reverse=True)][:3]
```




    [['RM_LSTAT', 0.7864889421124028],
     ['RM_TAX', 0.7750525123747647],
     ['RM_RAD', 0.7682152400234057]]



## Look at the top 3 interactions: "RM" as a confounding factor

The top three interactions seem to involve "RM", the number of rooms as a confounding variable for all of them. Let's have a look at interaction plots for all three of them. This exercise will involve:

- Splitting the data up in 3 groups: one for houses with a few rooms, one for houses with a "medium" amount of rooms, one for a high amount of rooms 
- Create a function `build_interaction_rm()`. This function takes an argument `varname` (which can be set equal to the column name as a string) and a column `description` (which describes the variable or varname, to be included on the x-axis of the plot). The function outputs a plot that uses "RM" as a confounding factor. Each plot should have three regression lines, one for each level of "RM"  

The data has been split into high, medium, and low number of rooms for you.


```python
rm = np.asarray(df[['RM']]).reshape(len(df[['RM']]))
```


```python
high_rm = all_data[rm > np.percentile(rm, 67)]
med_rm = all_data[(rm > np.percentile(rm, 33)) & (rm <= np.percentile(rm, 67))]
low_rm = all_data[rm <= np.percentile(rm, 33)]
```

Create `build_interaction_rm()`.


```python
def build_interaction_rm(varname, description):
    regression_1 = LinearRegression()
    regression_2 = LinearRegression()
    regression_3 = LinearRegression()

    high_1 = high_rm[varname].values.reshape(-1, 1)
    high_1_target = high_rm['target'].values.reshape(-1, 1)
    med_2 = med_rm[varname].values.reshape(-1, 1)
    med_2_target = med_rm['target'].values.reshape(-1, 1)
    low_3 = low_rm[varname].values.reshape(-1, 1)
    low_3_target = low_rm['target'].values.reshape(-1, 1)
    
    regression_1.fit(high_1, high_1_target)
    regression_2.fit(med_2, med_2_target)
    regression_3.fit(low_3, low_3_target)

    pred_1 = regression_1.predict(high_1)
    pred_2 = regression_2.predict(med_2)
    pred_3 = regression_3.predict(low_3)
    
    
    plt.figure(figsize=(10,6))
    plt.scatter(high_1, high_1_target,color='blue', alpha=0.3, label='high_rooms')
    plt.scatter(med_2,med_2_target, color='red', alpha=0.3, label='med_rooms')
    plt.scatter(low_3,low_3_target, color='orange', alpha=0.3, label='low_rooms')
    
    plt.plot(high_1, pred_1, color='blue')
    plt.plot(med_2, pred_2, color='red')
    plt.plot(low_3, pred_3, color='orange')
    plt.xlabel(description)
    plt.legend;
    
    pass
```

Next, use `build_interaction_rm()` with the three variables that came out with the highest effect on $R^2$. 


```python
# first plot
build_interaction_rm('LSTAT', 'lower status of the population')
```


![png](index_files/index_27_0.png)



```python
# second plot
build_interaction_rm('TAX', 'full-value property-tax rate per $10,000')
```


![png](index_files/index_28_0.png)



```python
# third plot
build_interaction_rm('RAD', 'index of accessibility to radial highways')
```


![png](index_files/index_29_0.png)


## Build a final model including all three interactions at once

Use 10-fold cross-validation to build a model using all the above interactions. 


```python
# code here
cross_validation = KFold(n_splits=10, shuffle=True, random_state=1)
all_data_interaction = all_data.drop(columns=['Target']).copy()
all_data_interaction['RM_LSTAT'] = all_data['RM'] * all_data['LSTAT']
all_data_interaction['RM_TAX'] = all_data['RM'] * all_data['TAX']
all_data_interaction['RM_RAD'] = all_data['RM'] * all_data['RAD']
```


```python
all_data_interaction.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>RM_LSTAT</th>
      <th>RM_TAX</th>
      <th>RM_RAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>32.74350</td>
      <td>1946.200</td>
      <td>6.575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>58.68794</td>
      <td>1553.882</td>
      <td>12.842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>28.95555</td>
      <td>1738.770</td>
      <td>14.370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>20.57412</td>
      <td>1553.556</td>
      <td>20.994</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>38.09351</td>
      <td>1586.634</td>
      <td>21.441</td>
    </tr>
  </tbody>
</table>
</div>




```python
# code here
regression_interaction = LinearRegression()
model_interaction_r2 = np.mean(cross_val_score(regression_interaction,
                                               all_data_interaction,y,scoring='r2', cv=cross_validation))
model_interaction_r2
```




    0.7852890964511973



Our $R^2$ has increased considerably! Let's have a look in `statsmodels` to see if all these interactions are significant.


```python
import statsmodels.api as sm
all_data_interaction = sm.add_constant(all_data_interaction)
model = sm.OLS(y,all_data_interaction)
results = model.fit()

results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Target</td>      <th>  R-squared:         </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.809</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   134.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 13 Apr 2020</td> <th>  Prob (F-statistic):</th> <td>3.25e-167</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:49:42</td>     <th>  Log-Likelihood:    </th> <td> -1413.9</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2862.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   489</td>      <th>  BIC:               </th> <td>   2934.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    16</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>    <td>  -14.8453</td> <td>    7.428</td> <td>   -1.998</td> <td> 0.046</td> <td>  -29.441</td> <td>   -0.250</td>
</tr>
<tr>
  <th>CRIM</th>     <td>   -0.1628</td> <td>    0.028</td> <td>   -5.756</td> <td> 0.000</td> <td>   -0.218</td> <td>   -0.107</td>
</tr>
<tr>
  <th>ZN</th>       <td>    0.0174</td> <td>    0.012</td> <td>    1.463</td> <td> 0.144</td> <td>   -0.006</td> <td>    0.041</td>
</tr>
<tr>
  <th>INDUS</th>    <td>    0.0900</td> <td>    0.053</td> <td>    1.707</td> <td> 0.088</td> <td>   -0.014</td> <td>    0.194</td>
</tr>
<tr>
  <th>CHAS</th>     <td>    2.5988</td> <td>    0.740</td> <td>    3.511</td> <td> 0.000</td> <td>    1.144</td> <td>    4.053</td>
</tr>
<tr>
  <th>NOX</th>      <td>  -13.4647</td> <td>    3.277</td> <td>   -4.109</td> <td> 0.000</td> <td>  -19.903</td> <td>   -7.026</td>
</tr>
<tr>
  <th>RM</th>       <td>   10.8250</td> <td>    0.986</td> <td>   10.976</td> <td> 0.000</td> <td>    8.887</td> <td>   12.763</td>
</tr>
<tr>
  <th>AGE</th>      <td>    0.0052</td> <td>    0.011</td> <td>    0.461</td> <td> 0.645</td> <td>   -0.017</td> <td>    0.028</td>
</tr>
<tr>
  <th>DIS</th>      <td>   -0.9547</td> <td>    0.175</td> <td>   -5.469</td> <td> 0.000</td> <td>   -1.298</td> <td>   -0.612</td>
</tr>
<tr>
  <th>RAD</th>      <td>    0.7093</td> <td>    0.476</td> <td>    1.489</td> <td> 0.137</td> <td>   -0.227</td> <td>    1.645</td>
</tr>
<tr>
  <th>TAX</th>      <td>    0.0333</td> <td>    0.025</td> <td>    1.354</td> <td> 0.176</td> <td>   -0.015</td> <td>    0.082</td>
</tr>
<tr>
  <th>PTRATIO</th>  <td>   -0.6849</td> <td>    0.113</td> <td>   -6.068</td> <td> 0.000</td> <td>   -0.907</td> <td>   -0.463</td>
</tr>
<tr>
  <th>B</th>        <td>    0.0048</td> <td>    0.002</td> <td>    2.068</td> <td> 0.039</td> <td>    0.000</td> <td>    0.009</td>
</tr>
<tr>
  <th>LSTAT</th>    <td>    1.1528</td> <td>    0.232</td> <td>    4.973</td> <td> 0.000</td> <td>    0.697</td> <td>    1.608</td>
</tr>
<tr>
  <th>RM_LSTAT</th> <td>   -0.2916</td> <td>    0.041</td> <td>   -7.169</td> <td> 0.000</td> <td>   -0.372</td> <td>   -0.212</td>
</tr>
<tr>
  <th>RM_TAX</th>   <td>   -0.0072</td> <td>    0.004</td> <td>   -1.828</td> <td> 0.068</td> <td>   -0.015</td> <td>    0.001</td>
</tr>
<tr>
  <th>RM_RAD</th>   <td>   -0.0699</td> <td>    0.078</td> <td>   -0.896</td> <td> 0.371</td> <td>   -0.223</td> <td>    0.083</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>255.231</td> <th>  Durbin-Watson:     </th> <td>   1.087</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2564.486</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.963</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.307</td>  <th>  Cond. No.          </th> <td>1.18e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.18e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# code here from statsmodels.formula.api import ols
from statsmodels.formula.api import ols
# SAME MODEL
predictors = all_data_interaction.columns
all_data_interaction['Target'] = all_data['Target']
outcome = 'Target'
f = '+'.join(predictors)
formula = outcome + '~' + f
stats_model = ols(formula=formula, data=all_data_interaction).fit()
```


```python
stats_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Target</td>      <th>  R-squared:         </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.809</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   134.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 13 Apr 2020</td> <th>  Prob (F-statistic):</th> <td>3.25e-167</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:50:36</td>     <th>  Log-Likelihood:    </th> <td> -1413.9</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2862.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   489</td>      <th>  BIC:               </th> <td>   2934.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    16</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -7.4227</td> <td>    3.714</td> <td>   -1.998</td> <td> 0.046</td> <td>  -14.720</td> <td>   -0.125</td>
</tr>
<tr>
  <th>const</th>     <td>   -7.4227</td> <td>    3.714</td> <td>   -1.998</td> <td> 0.046</td> <td>  -14.720</td> <td>   -0.125</td>
</tr>
<tr>
  <th>CRIM</th>      <td>   -0.1628</td> <td>    0.028</td> <td>   -5.756</td> <td> 0.000</td> <td>   -0.218</td> <td>   -0.107</td>
</tr>
<tr>
  <th>ZN</th>        <td>    0.0174</td> <td>    0.012</td> <td>    1.463</td> <td> 0.144</td> <td>   -0.006</td> <td>    0.041</td>
</tr>
<tr>
  <th>INDUS</th>     <td>    0.0900</td> <td>    0.053</td> <td>    1.707</td> <td> 0.088</td> <td>   -0.014</td> <td>    0.194</td>
</tr>
<tr>
  <th>CHAS</th>      <td>    2.5988</td> <td>    0.740</td> <td>    3.511</td> <td> 0.000</td> <td>    1.144</td> <td>    4.053</td>
</tr>
<tr>
  <th>NOX</th>       <td>  -13.4647</td> <td>    3.277</td> <td>   -4.109</td> <td> 0.000</td> <td>  -19.903</td> <td>   -7.026</td>
</tr>
<tr>
  <th>RM</th>        <td>   10.8250</td> <td>    0.986</td> <td>   10.976</td> <td> 0.000</td> <td>    8.887</td> <td>   12.763</td>
</tr>
<tr>
  <th>AGE</th>       <td>    0.0052</td> <td>    0.011</td> <td>    0.461</td> <td> 0.645</td> <td>   -0.017</td> <td>    0.028</td>
</tr>
<tr>
  <th>DIS</th>       <td>   -0.9547</td> <td>    0.175</td> <td>   -5.469</td> <td> 0.000</td> <td>   -1.298</td> <td>   -0.612</td>
</tr>
<tr>
  <th>RAD</th>       <td>    0.7093</td> <td>    0.476</td> <td>    1.489</td> <td> 0.137</td> <td>   -0.227</td> <td>    1.645</td>
</tr>
<tr>
  <th>TAX</th>       <td>    0.0333</td> <td>    0.025</td> <td>    1.354</td> <td> 0.176</td> <td>   -0.015</td> <td>    0.082</td>
</tr>
<tr>
  <th>PTRATIO</th>   <td>   -0.6849</td> <td>    0.113</td> <td>   -6.068</td> <td> 0.000</td> <td>   -0.907</td> <td>   -0.463</td>
</tr>
<tr>
  <th>B</th>         <td>    0.0048</td> <td>    0.002</td> <td>    2.068</td> <td> 0.039</td> <td>    0.000</td> <td>    0.009</td>
</tr>
<tr>
  <th>LSTAT</th>     <td>    1.1528</td> <td>    0.232</td> <td>    4.973</td> <td> 0.000</td> <td>    0.697</td> <td>    1.608</td>
</tr>
<tr>
  <th>RM_LSTAT</th>  <td>   -0.2916</td> <td>    0.041</td> <td>   -7.169</td> <td> 0.000</td> <td>   -0.372</td> <td>   -0.212</td>
</tr>
<tr>
  <th>RM_TAX</th>    <td>   -0.0072</td> <td>    0.004</td> <td>   -1.828</td> <td> 0.068</td> <td>   -0.015</td> <td>    0.001</td>
</tr>
<tr>
  <th>RM_RAD</th>    <td>   -0.0699</td> <td>    0.078</td> <td>   -0.896</td> <td> 0.371</td> <td>   -0.223</td> <td>    0.083</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>255.231</td> <th>  Durbin-Watson:     </th> <td>   1.087</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2564.486</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.963</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.307</td>  <th>  Cond. No.          </th> <td>5.99e+17</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.09e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



What is your conclusion here?


```python
# formulate your conclusion
```

Only the interaction between number of rooms and lower status of population (LSTAT) are significant in predicting housing prices.

## Summary

You should now understand how to include interaction effects in your model! As you can see, interactions can have a strong impact on linear regression models, and they should always be considered when you are constructing your models.
