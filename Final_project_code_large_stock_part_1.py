import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from scipy.stats import linregress

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load all common stocks
file_path = "strdata.csv"
data = pd.read_csv(file_path, header=0, low_memory=False)

# Sort by date
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data = data.sort_values(by='date')

# Identify and quantify errors
errors = ['NaN', "B", "C", "-66", "-77", "-99"]
error_counts = {err: np.sum(data['RET'].astype(str).isin(errors)) for err in errors}
print("Error counts:", error_counts)

# Replace invalid values in return column with -0.5 before converting to float
data['RET'] = np.where(data['RET'].astype(str).isin(errors), -0.5, data['RET']).astype(float)

# Filter penny stocks since 2021
cutoff_date = datetime(2021, 1, 1)
data = data[data['date'] >= cutoff_date]
data['avg_price'] = (data['BID'].astype(float) + data['ASK'].astype(float)) / 2
penny_stock_mask = (data['avg_price'] < 5)
bad_permnos = data[penny_stock_mask]['PERMNO'].unique()
data = data[~data['PERMNO'].isin(bad_permnos)]

# Import FF5 Excel data
def excel_to_2d_array(filename, sheet_name):
    df = pd.read_excel(filename, sheet_name=sheet_name)
    headers = df.columns.tolist()
    data = df.values.tolist()
    return [headers] + data

filename = 'FF5_example1.xlsx'
sheet_name = 'FF5'
FF5_raw = excel_to_2d_array(filename, sheet_name)

# Add Mkt-RF and RF to data
FF5 = np.array(FF5_raw[1:], dtype=object)
FF5[:, 0] = [str(int(float(d))).zfill(8) for d in FF5[:, 0]]
data_dates_formatted = np.array([d.replace('-', '') for d in data['date'].dt.strftime('%Y-%m-%d')])
ff5_dict = {row[0]: (float(row[1]), float(row[2])) for row in FF5}
ff5_dates = np.array(list(ff5_dict.keys()))
ff5_mktrf = np.array([ff5_dict[d][0] for d in ff5_dates])
ff5_rf = np.array([ff5_dict[d][1] for d in ff5_dates])
match_idx = np.searchsorted(ff5_dates, data_dates_formatted)
mkt_rf_col = np.full(len(data), np.nan)
excess_ret_col = np.full(len(data), np.nan)
valid_mask = (match_idx < len(ff5_dates)) & (ff5_dates[match_idx] == data_dates_formatted)
mkt_rf_col[valid_mask] = ff5_mktrf[match_idx[valid_mask]]
excess_ret_col[valid_mask] = data['RET'].values[valid_mask] - ff5_rf[match_idx[valid_mask]]
data['MktRF'] = mkt_rf_col
data['ExcessRet'] = excess_ret_col

# Add Month-Year column
data['Month'] = data['date'].dt.strftime('%m/%Y')

df = data[['PERMNO', 'Month', 'MktRF', 'ExcessRet']].dropna(subset=['MktRF', 'ExcessRet'])

# Function to compute regression, monthly excess return, and monthly std dev
def get_regression_stats(group):
    if len(group) > 1:
        if np.all(group['MktRF'] == group['MktRF'].iloc[0]):
            return pd.Series({'Alpha': np.nan, 'Beta': np.nan, 'MonthlyAvgExcessRet': np.nan, 'MonthlyStdDevExcessRet': np.nan})
        slope, intercept, *_ = linregress(group['MktRF'], group['ExcessRet'])
        avg_daily_excess = group['ExcessRet'].mean()
        std_daily_excess = group['ExcessRet'].std()
        avg_monthly_excess = avg_daily_excess * 21
        std_monthly_excess = std_daily_excess * np.sqrt(21)
        return pd.Series({
            'Alpha': intercept,
            'Beta': slope,
            'MonthlyAvgExcessRet': avg_monthly_excess,
            'MonthlyStdDevExcessRet': std_monthly_excess
        })
    else:
        return pd.Series({'Alpha': np.nan, 'Beta': np.nan, 'MonthlyAvgExcessRet': np.nan, 'MonthlyStdDevExcessRet': np.nan})

lin_reg_df = df.groupby(['PERMNO', 'Month']).apply(get_regression_stats).dropna().reset_index()

# Filter out stocks with -0.5 return
lin_reg_df = lin_reg_df[lin_reg_df['MonthlyAvgExcessRet'] != -0.5]

# Calculate Monthly Average of BID and ASK for each stock (PERMNO) and month
monthly_avg_prices = data.groupby(['PERMNO', 'Month'])[['BID', 'ASK']].mean().reset_index()
monthly_avg_prices['Price'] = (monthly_avg_prices['BID'] + monthly_avg_prices['ASK']) / 2

monthly_price_data = pd.merge(lin_reg_df, monthly_avg_prices[['PERMNO', 'Month', 'Price']], on=['PERMNO', 'Month'], how='left')

# Preview result
print(monthly_price_data.head())

# Export to CSV
monthly_price_data.to_csv('monthly_price_data.csv', index=False)
