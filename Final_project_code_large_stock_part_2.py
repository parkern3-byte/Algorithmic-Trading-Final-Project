import pandas as pd
import numpy as np

# Load data
print("Largest Stock Results")
file_path = "monthly_price_data.csv"
data = pd.read_csv(file_path, header=0, low_memory=False)
print(data.head())
data['Month_dt'] = pd.to_datetime(data['Month'], format='%m/%Y', errors='coerce')
data = data.dropna(subset=['Month_dt'])

# Sort by PERMNO and Month
data = data.sort_values(['PERMNO', 'Month_dt']).reset_index(drop=True)

# Group by PERMNO
grouped = data.groupby('PERMNO')

# Shift columns to simulate the lag effect
data['Alpha'] = grouped['Alpha'].shift(1)
data['Beta'] = grouped['Beta'].shift(1)
data['MonthlyAvgExcessRet'] = grouped['MonthlyAvgExcessRet'].shift(1)
data['MonthlyStdDevExcessRet'] = grouped['MonthlyStdDevExcessRet'].shift(1)

# Drop rows with NaN values after shifting
data = data.dropna(subset=['Alpha', 'Beta', 'MonthlyAvgExcessRet', 'MonthlyStdDevExcessRet'])
data.loc[:, 'Month'] = data['Month_dt'].dt.strftime('%b-%y')
stocks_per_month = data.groupby('Month')['PERMNO'].nunique()

# Filter top 200 stocks based on 'Price' for each month
top_200_stocks_per_month = data.sort_values(by=['Month', 'Price'], ascending=[True, False]) \
    .groupby('Month').head(200)

if len(top_200_stocks_per_month) == 0:
    print("No stocks found after filtering. Please check the filtering criteria.")

# Find the monthly average returns of the portfolio
monthly_avg_returns = top_200_stocks_per_month.groupby('Month')['MonthlyAvgExcessRet'].mean()

# Check if the monthly_avg_returns is empty
if monthly_avg_returns.empty:
    print("Error: monthly_avg_returns is empty")
else:
    E = monthly_avg_returns.to_numpy()

    # Apply 1% transaction costs
    transaction_cost_pct = 0.01 
    E_adjusted = E - (E * transaction_cost_pct)
    monthly_volatility_array = []

    #correlation coefficient to 0.3
    correlation_coefficient = 0.3

    for month in top_200_stocks_per_month['Month'].unique():
        month_data = top_200_stocks_per_month[top_200_stocks_per_month['Month'] == month]
        monthly_stdevs = month_data['MonthlyStdDevExcessRet'].values

        correlation_matrix = np.full((len(monthly_stdevs), len(monthly_stdevs)), correlation_coefficient)
        np.fill_diagonal(correlation_matrix, 1)

        covariance_matrix = np.outer(monthly_stdevs, monthly_stdevs) * correlation_matrix
        weights = np.ones(len(covariance_matrix)) / len(covariance_matrix)

        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        monthly_volatility_array.append(portfolio_volatility)

    monthly_volatility_array = np.array(monthly_volatility_array)

    # Display results
    print("\nMonthly Portfolio Volatility (Standard Deviation):")
    print(monthly_volatility_array)

    # Cumulative return
    cumulative_return = np.prod(1 + E_adjusted)
    print(f"\nOverall Cumulative Return: {cumulative_return:.2f}")

    # Number of months in the return series
    num_months = len(E_adjusted)
    if num_months > 0:
        annualized_return = (cumulative_return ** (12 / num_months)) - 1
        print(f"\nAnnualized Return: {annualized_return:.4%}")
    else:
        print("Error: No months found for calculation.")

    # Calculate average monthly volatility
    avg_monthly_volatility = np.mean(monthly_volatility_array)
    annualized_volatility = avg_monthly_volatility * np.sqrt(12)
    print(f"\nAnnualized Volatility of the Portfolio: {annualized_volatility:.4%}")

    # Max Drawdown
    combined_monthly_return = np.array(E_adjusted).cumsum() + 1
    HWM = [1]
    DDt = [0]
    for i in range(1, len(E_adjusted)):
        HWM.append(combined_monthly_return[i])
        HWM[i] = max(HWM)
        DDt.append(100 * (HWM[i] - combined_monthly_return[i]))
    print("The maximum Drawdown in the period is {0:0.3f}% \n".format(max(DDt)))

    # Win Rate
    num_positive_months = np.sum(E_adjusted > 0)
    monthly_win_rate = num_positive_months / len(E_adjusted) * 100
    print(f"Monthly Win Rate: {monthly_win_rate:.2f}%")

    # Months
    months_array = data['Month'].unique()
    print("\nArray of Months:")
    print(months_array)

    # Annualized Sharpe Ratio
    rf = 0.02076239
    SR = (annualized_return - rf) / annualized_volatility
    print("The Sharpe Ratio is {0:0.3f}  \n".format(SR))
