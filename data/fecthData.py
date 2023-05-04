import re

import baostock as bs
import pandas as pd
from stockLists import *

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

for mode in ('Train','Test'):
    if mode=='Train':
        start_time, end_time = train_time_range
    else:
        start_time, end_time = test_time_range

    for ind in stock_d:
        print(mode,"qcuring stock:",ind,stock_d[ind])
        #### 获取沪深A股历史K线数据 ####
        # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        rs = bs.query_history_k_data_plus(ind,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_time, end_date=end_time,
            frequency="d", adjustflag="1")
        if rs.error_code!=0:
            print('query_history_k_data_plus respond error_code:'+rs.error_code)
            print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        else:
            print("download successful!")

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') and rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 结果集输出到csv文件 ####
        result.to_csv(mode+stock_d[ind]+".csv", index=False)

#### 登出系统 ####
bs.logout()