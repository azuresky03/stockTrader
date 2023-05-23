import re

import baostock as bs
import pandas as pd
from stockLists import *

#! run command "pip install baostock before running this script"
#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

for mode in ('Train','Test'):
    if mode == 'Train':
        start_time, end_time = train_time_range
    else:
        start_time, end_time = test_time_range

    #stock code is id for stock like 'sz.002594', stock_d[stock_code] is name like '比亚迪'
    for stock_code in stock_d:
        print(mode,"qcuring stock:",stock_code,stock_d[stock_code])
        #### 获取沪深A股历史K线数据 ####
        # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        rs = bs.query_history_k_data_plus(stock_code,
            #the data to acquire
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_time, end_date=end_time,
            frequency="d", adjustflag="1")
        if rs.error_code!='0':
            print('query_history_k_data_plus respond error_code:'+rs.error_code)
            print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        else:
            print("download successful!")

        # create pd.DataFrame object using data from rs 

        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())

        result = pd.DataFrame(data_list, columns=rs.fields)

        ''' 
        # TODO : Calculate following values and append them as extra column in csv files.
        #? maybe use pandas library
        1. MACD 
        2. RSI
        3. ADX
        '''

        #### 结果集输出到csv文件 ####
        # each stock have seperate csv files.
        result.to_csv(mode+stock_d[stock_code]+".csv", index=False)

#### 登出系统 ####
bs.logout()