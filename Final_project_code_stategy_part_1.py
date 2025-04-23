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
data = data.to_numpy()

# Sort by date
data = sorted(data, key=lambda row: datetime.strptime(row[1], '%Y-%m-%d'))
data = np.array(data)

# Identify and quantify errors
errors = ['NaN', "B", "C", "-66", "-77", "-99"]
col5_str = data[:, 5].astype(str)
error_counts = {err: np.sum(col5_str == err) for err in errors}
print("Error counts:", error_counts)

# Replace invalid values in return column with -0.5 before converting to float
data[:, 5] = np.where(np.isin(col5_str, errors), -0.5, data[:, 5])
data[:, 5] = data[:, 5].astype(float)

# Sort again after cleaning
data = sorted(data, key=lambda row: datetime.strptime(row[1], '%Y-%m-%d'))
data = np.array(data)

# Filter penny stocks since 2021
date_col = np.array([datetime.strptime(d, '%Y-%m-%d') for d in data[:, 1]])
cutoff_date = datetime(2021, 1, 1)
data = data[date_col >= cutoff_date]

#calculate the average price and filter penny stocks
avg_price = (data[:, 6].astype(float) + data[:, 7].astype(float)) / 2
penny_stock_mask = (avg_price < 5)
bad_permnos = np.unique(data[penny_stock_mask, 0])
data = data[~np.isin(data[:, 0], bad_permnos)]

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
data_dates_formatted = np.array([d.replace('-', '') for d in data[:, 1]])
ff5_dict = {row[0]: (float(row[1]), float(row[2])) for row in FF5}
ff5_dates = np.array(list(ff5_dict.keys()))
ff5_mktrf = np.array([ff5_dict[d][0] for d in ff5_dates])
ff5_rf = np.array([ff5_dict[d][1] for d in ff5_dates])
match_idx = np.searchsorted(ff5_dates, data_dates_formatted)
mkt_rf_col = np.full(len(data), np.nan)
excess_ret_col = np.full(len(data), np.nan)
valid_mask = (match_idx < len(ff5_dates)) & (ff5_dates[match_idx] == data_dates_formatted)
mkt_rf_col[valid_mask] = ff5_mktrf[match_idx[valid_mask]]
excess_ret_col[valid_mask] = data[valid_mask, 5] - ff5_rf[match_idx[valid_mask]]
data = np.hstack((data, mkt_rf_col.reshape(-1, 1), excess_ret_col.reshape(-1, 1)))

# Add Month-Year column
data_dates = np.array([datetime.strptime(d, '%Y-%m-%d') for d in data[:, 1]])
month_year = [d.strftime('%Y-%m') for d in data_dates]
data = np.hstack((data, np.array(month_year).reshape(-1, 1)))

# Convert relevant portion to a DataFrame for groupby analysis
df = pd.DataFrame({
    'PERMNO': data[:, 0],
    'Month': data[:, -1],
    'MktRF': data[:, -3].astype(float),
    'ExcessRet': data[:, -2].astype(float),
})

# Drop missing values
df = df.dropna(subset=['MktRF', 'ExcessRet'])

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

# Shift Alpha and Beta forward by 1 month to avoid look-ahead bias
lin_reg_df['Month'] = pd.to_datetime(lin_reg_df['Month'])
lin_reg_df['NextMonth'] = lin_reg_df['Month'] + pd.DateOffset(months=1)
lin_reg_df['NextMonth'] = lin_reg_df['NextMonth'].dt.strftime('%Y-%m')

# Shift values to next month
lin_reg_df_shifted = lin_reg_df.copy()
lin_reg_df_shifted['Month'] = lin_reg_df_shifted['NextMonth']
lin_reg_df_shifted = lin_reg_df_shifted.drop(columns='NextMonth')

# export to new csv
lin_reg_df_shifted = lin_reg_df_shifted.dropna()
lin_reg_df_shifted.to_csv('lin_reg_results_monthly.csv', index=False)

print("done")
