import cvxpy as cp
import numpy as np
import pandas as pd
#initialize Alpaca Client
from alpaca.data.historical import StockHistoricalDataClient
key='PK9UUD22E5HQQF4RHDFD'
secret='PahVpPcGMR9uylE8Lvcip83z3zytN7NlqZ709qYk'
client = StockHistoricalDataClient(key,secret)

#dates
from datetime import datetime
start_date=datetime(2020,1,1)
end_date=datetime(2023,12,31)
symbols=sorted(['AAPL','MSFT','TSLA','GOOGL','NVDA','META','AMZN'])
returns=[]
n=0
ones=[] 
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
    stock_df['Daily return'] = stock_df['close'].pct_change()
    stock_df['Cumulative Return'] = (1 + stock_df['Daily return']).cumprod()
    pd.set_option('display.max_columns', None)

    #convert stock data into 2d array
    import numpy as np
    stock_data= stock_df.reset_index().to_numpy()

    #create 2d array of daily returns
    returns.append([])
    stock_data[0][9]=0
    for i in range(0,len(stock_data)):
        returns[n].append(stock_data[i][9])
    ones.append(1)
    n=n+1

#transpose daily returns matrix
returns=np.array(returns)
returns=returns.T

#find Covariance matrix
cov_matrix=np.cov(returns,rowvar=False)
n=len(cov_matrix)

Lambda=[0,0.1,0.5,1,5]
W_total=[]
i=0
for y in Lambda:
    print("λ= {0}\n".format(y)) 
    #define variables
    weights=cp.Variable(n) #creates n-dimesnional vector, n is number of assets

    #define objective function-minimize variance of portfolio
    risk=cp.quad_form(weights,cov_matrix)+y*cp.sum(cp.abs(weights)) #W^T@sigma@W
    objective=cp.Minimize(risk) #or cp.Maximize(...)

    #define constraints
    constraints=[
        cp.sum(weights)==1,
        #...
    ]

    #create and colve problem
    problem=cp.Problem(objective, constraints)
    result=problem.solve()

    #Access solution
    optimal_value=problem.value

    if problem.status != cp.OPTIMAL:
            raise Exception(f"Optimization failed with status: {problem.status}")
    else:
        print("Optimal Portfolio Weights")
        optimal_weights = weights.value
        optimal_weights[(optimal_weights < 1e-6)&(optimal_weights > -1e-6)]=0
        optimal_weights=optimal_weights.tolist()
        for i in range(0,len(symbols)):
            print("{0} {1:0.3f}  \n".format(symbols[i],optimal_weights[i]))
        W_total.append(optimal_weights)
        i=i+1





