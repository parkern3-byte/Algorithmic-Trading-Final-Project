import pandas as pd
import numpy as np
import warnings
from datetime import datetime

print("Strategy Results")

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load data
file_path = "lin_reg_results_monthly.csv"
data = pd.read_csv(file_path, header=0, low_memory=False)

# Convert 'Month' column to datetime
data["Month_dt"] = pd.to_datetime(data["Month"], format="%Y-%m")

# Sort data
data = data.sort_values(["PERMNO", "Month_dt"]).reset_index(drop=True)

# Shift Alpha, Beta, and return metrics back by one month
shifted_data = data.copy()
grouped = data.groupby("PERMNO")
shifted_data["Alpha"] = grouped["Alpha"].shift(1)
shifted_data["Beta"] = grouped["Beta"].shift(1)
shifted_data["MonthlyAvgExcessRet"] = grouped["MonthlyAvgExcessRet"].shift(1)
shifted_data["MonthlyStdDevExcessRet"] = grouped["MonthlyStdDevExcessRet"].shift(1)

# Drop rows with missing values
shifted_data = shifted_data.dropna(subset=["Alpha", "Beta", "MonthlyAvgExcessRet", "MonthlyStdDevExcessRet"])

# Apply Beta constraint to lagged Beta
shifted_data = shifted_data[
    (shifted_data["Beta"] >= -0.2) & 
    (shifted_data["Beta"] <= 0.2)
]

# Re-format Month column
shifted_data["Month"] = shifted_data["Month_dt"].dt.strftime("%Y-%m")

# Select top 200 by Alpha per month
top200_by_month = (
    shifted_data
    .groupby("Month", group_keys=False)
    .apply(lambda group: group.nlargest(200, "Alpha"))
    .reset_index(drop=True)
)

# Remove January 2021 from signal months
top200_by_month = top200_by_month[top200_by_month["Month"] != "2021-01"]
top200_by_month = top200_by_month.sort_values(by="Month_dt").drop(columns=["Month_dt"])

# Number of stocks in each month's portfolio
stocks_per_month = top200_by_month.groupby("Month")["PERMNO"].nunique().reset_index(name="Number of Stocks")

# Calculate average monthly returns
monthly_avg_returns = top200_by_month.groupby("Month")["MonthlyAvgExcessRet"].mean()

# Convert returns to numpy array
E = monthly_avg_returns.to_numpy()

# Apply 1% trading cost
transaction_cost_pct = 0.01
E_adjusted = E - (E * transaction_cost_pct)

# Calculate monthly portfolio volatility
monthly_volatility_array = []

correlation_coefficient = 0.3

for month in top200_by_month["Month"].unique():
    month_data = top200_by_month[top200_by_month["Month"] == month]
    monthly_stdevs = month_data["MonthlyStdDevExcessRet"].values
    correlation_matrix = np.full((len(monthly_stdevs), len(monthly_stdevs)), correlation_coefficient)
    np.fill_diagonal(correlation_matrix, 1)
    covariance_matrix = np.outer(monthly_stdevs, monthly_stdevs) * correlation_matrix
    weights = np.ones(len(covariance_matrix)) / len(covariance_matrix)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    monthly_volatility_array.append(portfolio_volatility)

monthly_volatility_array = np.array(monthly_volatility_array)

# Calculate cumulative return
cumulative_return = np.prod(1 + E_adjusted)
print(f"\nOverall Cumulative Return: {cumulative_return:.2f}")

# Calculate annualized return
num_months = len(E_adjusted)
annualized_return = (cumulative_return ** (12 / num_months)) - 1
print(f"\nAnnualized Return: {annualized_return:.4%}")

# Average monthly volatility and annualized
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

# Monthly Win Rate
num_positive_months = np.sum(E_adjusted > 0)
monthly_win_rate = num_positive_months / len(E_adjusted) * 100
print(f"Monthly Win Rate: {monthly_win_rate:.2f}%")

# Display array of months
months_array = shifted_data["Month"].unique()
print("\nArray of Months:")
print(months_array)

# Annualized Sharpe Ratio
# Annualized Rf from 2021-2023 from FF5
rf = 0.02076239
SR = (annualized_return - rf) / annualized_volatility
print(f"The Sharpe Ratio is {SR:.3f}  \n")
