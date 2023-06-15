# stockTrader
- environment requirement:  
python 3.10  
pacakges:  
gym==0.21.0, stable_baselines3==1.8.0  
(Not required: FinRL, akshare, baostock)

- [tutorial.ipynb](https://github.com/azuresky03/stockTrader/blob/master/tutorial.ipynb) is a guide of this project works  

- project arrangement  
model : contains all the processed data and scripts  
&nbsp;&nbsp;&nbsp;|---direct : contains train&test data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---train.csv  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---test.csv  
&nbsp;&nbsp;&nbsp;|---model-env2 : utilzing version 2 environment  
&nbsp;&nbsp;&nbsp;|---model-env3 : utilzing version 3 environment  
&nbsp;&nbsp;&nbsp;|---stock_env : our customized stock trading environment  
