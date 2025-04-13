#import market data from excel into 2d array
import pandas as pd
def  excel_to_2d_array(filename, sheet_name):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    headers = df.columns.tolist()
    data = df.values.tolist()
    return [headers] + data
filename='FF5_example1.xlsx'
sheet_name='FF5'
FF5=excel_to_2d_array(filename,sheet_name)

#initialize Alpaca Client
from alpaca.data.historical import StockHistoricalDataClient
key='PK9UUD22E5HQQF4RHDFD'
secret='PahVpPcGMR9uylE8Lvcip83z3zytN7NlqZ709qYk'
client = StockHistoricalDataClient(key,secret)

#dates
from datetime import datetime
end_date=datetime(2023,12,31)
start_date=datetime(2019,1,1)

symbols=['AAPL','MSFT','TSLA','GOOGL','NVDA','META','AMZN']
prices=[]
n=0
#loop through companies 
#fetch stock data
for symbol in symbols:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    request_params=StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment='split'
        )
    stock_bars=client.get_stock_bars(request_params)
    stock_df=stock_bars.df

    #convert stock data into 2d array
    import numpy as np
    stock_data=stock_df.to_numpy() #close price equals index 3

    #calculate return for each price
    Rt=[0]
    for i in range(1,len(stock_data)):
        Rt.append(stock_data[i][3]/stock_data[i-1][3]-1)
        
    #create arrays for Rt-rf and RtM-rf
    j=0
    RtM_rf=[]
    Rt_rf=[]
    for i in range(13846,15104):
        RtM_rf.append(FF5[i][1])
        Rt_rf.append(Rt[j]-FF5[i][2])
        j=j+1

    #linear regression
    import statsmodels.api as sm
    x_with_const = sm.add_constant(RtM_rf)
    model = sm.OLS(Rt_rf, x_with_const).fit()
    intercept = model.params[0]
    coefficient = model.params[1]
    print("{0} alpha={1:0.6f}, beta={2:0.6f} \n".format(symbol,intercept,coefficient))

      #2d array of price data to rebalance 
    prices.append([])
    for i in range(0,len(stock_data)):
        prices[n].append(stock_data[i][3])    
    n=n+1
#rebounding portfolio
value=[10000]
shares=[0,0,0,0,0,0,0]
ind=10000/7
ind_values=[ind,ind,ind,ind,ind,ind,ind]
time=[0]
#initial share values
for j in range(len(shares)):
        shares[j]=(value[0]/7)/prices[j][0]
        
#following share values
for i in range(1,len(prices[0])):
    for j in range(len(shares)):
        ind_values[j]=shares[j]*prices[j][i]
    value.append(sum(ind_values))
    ind=value[i]/7
    for j in range(len(shares)):
            shares[j]=shares[j]+((ind-ind_values[j])/prices[j][i])
    time.append(i)
end_value=value[len(value)-1]
overall_return=(end_value/10000-1)*100
print("final value of portfolio is ${0:0.2f} \n".format(end_value))
print("the cumulative return of the portfolio is {0:0.2f}% \n".format(overall_return))
#plot dollar value
import matplotlib.pyplot as plt            
plt.plot(time, value)
plt.xlabel("days")
plt.ylabel("doller value")
plt.title("portfolio value over time (2019-2023)")
plt.text(3,25, "NOTE: the x axis refers to days that the market is open", fontsize=12, color='red', ha='left')
plt.show()        
    

