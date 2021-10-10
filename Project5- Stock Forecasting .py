#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries & Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action="ignore", category= FutureWarning)
warnings.filterwarnings(action="ignore", category= DeprecationWarning)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


# In[2]:


dataparse= lambda dates:pd.datetime.strptime(dates, "%Y-%m-%d")


# In[3]:


df= pd.read_csv(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Stock forecasting project\gs.us.txt",
                sep=",", parse_dates=["Date"], index_col=["Date"], date_parser=dataparse)


# In[4]:


df.head()


# In[5]:


df.tail()


# # Exploratory Data Analysis

# In[6]:


plt.figure(figsize=(10,5))
plt.xlabel("Dates")
plt.ylabel("Open Prices")
plt.plot(df["Open"])
plt.grid(True)


# # Auto-correlation Plot

# In[7]:


values=pd.DataFrame(df["Open"].values)


# In[8]:


dataframe=pd.concat([values, values.shift(1), values.shift(5), values.shift(10), values.shift(30)], axis=1)
dataframe.columns= ["t", "t+1", "t+5", "t+10", "t+30"]
dataframe.head()


# In[9]:


result=dataframe.corr()
print(result)


# # Dickey-Fuller method to check Stationary & Seasonality

# In[10]:


def tsplot(y, lags=None, figsize=(12,8), style="bmh"):
    
    if not isinstance(y, pd.Series):
        y=pd.Series(y)
        
    with plt.style.context(style="bmh"):
        fig=plt.figure(figsize=figsize)
        layout=(2,2)
        
        ts_ax= plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax= plt.subplot2grid(layout, (1,0))
        pacf_ax= plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value=sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title("Time Series Analysis Plot\n Dickey-Fuller: p={0:.5f}".format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[11]:


tsplot(df.Open, lags=30)


# In[12]:


data= df.copy(deep=False)
data1= df.copy(deep=False)
data1.Open= data1.Open - data1.Open.shift(1)
tsplot(data1.Open[1:], lags=30)


# In[13]:


train_data, test_data= data[1:-73], data[-73:]
plt.figure(figsize=(12,8))
plt.xlabel=("Dates")
plt.ylabel("Open Prices")
plt.plot(data["Open"].tail(600), "green", label="Train_data")
plt.plot(test_data["Open"], "blue", label="Test_data")
plt.grid(True)
plt.legend()
plt.show()


# # Mean Value Plot

# In[14]:


mean_value= data["Open"].mean()
mean_value


# In[15]:


plt.figure(figsize=(10,8))
plt.ylabel("Open Prices")
plt.plot(data["Open"], "green", label="Train_Data")
plt.plot(test_data["Open"], "blue", label="Test_data")
plt.axhline(y=mean_value, xmin=0.85, xmax=1, color= "red")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,8))
plt.ylabel("Open Prices")
plt.plot(data["Open"].tail(600), "green", label="Train_data")
plt.plot(test_data["Open"], "blue", label="Test_data")
plt.axhline(y=mean_value, xmin=0.85, xmax=1, color= "red")
plt.grid(True)
plt.legend()


# In[16]:


print("MSE:" + str(mean_squared_error(test_data["Open"], np.full(len(test_data), mean_value))))
print("MAE:" + str(mean_absolute_error(test_data["Open"], np.full(len(test_data), mean_value))))
print("RMSE:" + str(sqrt(mean_squared_error(test_data["Open"], np.full(len(test_data), mean_value)))))


# # Model Building & Validation

# ### Auto-Regressive Mode

# In[17]:


train_ar= train_data["Open"]
test_ar=test_data["Open"]

model=AR(train_ar)
model_fit= model.fit()


# In[18]:


window= model_fit.k_ar
coef= model_fit.params


# In[19]:


coef


# In[20]:


window


# In[21]:


history= train_ar[len(train_ar) - window:]
history= [history[i] for i in range(len(history))]


# In[22]:


predictions= list()
for t in range(len(test_ar)):
    length= len(history)
    lag= [history[i] for i in range(length-window,length)]
    yhat= coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    
    obs=test_ar[t]
    predictions.append(yhat)
    history.append(obs)


# In[23]:


plt.figure(figsize=(10,8))
plt.plot(data.index[-600:], data["Open"].tail(600), color="green", label= "Close price")
plt.plot(test_data.index, test_data["Open"], color= "red", label= "Test Close Price")
plt.plot(test_data.index, predictions, color="blue", label= "Predict Close Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
print("Lag: %s" % model_fit.k_ar)
plt.plot(data.index[-100:], data["Open"].tail(100), color="green", label= "Close price")
plt.plot(test_data.index, test_data["Open"], color= "red", label= "Test Close Price")
plt.plot(test_data.index, predictions, color="blue", label= "Predict Close Price")
plt.grid(True)
plt.legend()
plt.show()


# In[25]:


print("MSE:" + str(mean_squared_error(test_data["Open"], predictions)))
print("MAE:" + str(mean_absolute_error(test_data["Open"], predictions)))
print("RMSE:" + str(sqrt(mean_squared_error(test_data["Open"], predictions))))


# # Moving Average Model

# In[26]:


train_ma= train_data["Open"]
test_ma= test_data["Open"]

history= [x for x in train_ma]
y= test_ma

predictions= list()
model= ARMA(history, order=(0,8))
model_fit= model.fit(disp=0)
yhat= model_fit.forecast()[0]
predictions.append(yhat)


# In[27]:


history.append(y[0])
for i in range(1, len(y)):
    model= ARMA(history, order=(0,8))
    model_fit= model.fit(disp=0)
    yhat= model_fit.forecast()[0]
    predictions.append(yhat)
    obs= y[i]
    history.append(obs)


# In[28]:


plt.figure(figsize=(10,8))
plt.plot(data.index[-600:], data["Open"].tail(600), color="green", label= "Train Stock Price")
plt.plot(test_data.index, y, color= "red", label= "Real Close Price")
plt.plot(test_data.index, predictions, color="blue", label= "Predict Stock Price")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(data.index[-100:], data["Open"].tail(100), color="green", label= "Train Stcok price")
plt.plot(test_data.index, y, color= "red", label= "Real Close Price")
plt.plot(test_data.index, predictions, color="blue", label= "Predict Close Price")
plt.grid(True)
plt.legend()
plt.show()


# In[29]:


print("MSE:" + str(mean_squared_error(y, predictions)))
print("MAE:" + str(mean_absolute_error(y, predictions)))
print("RMSE:" + str(sqrt(mean_squared_error(y, predictions))))


# In[ ]:




