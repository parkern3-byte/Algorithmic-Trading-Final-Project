import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

sys.stdout = open(os.devnull, 'w')

# Import variables from last scripts
from Final_project_code_stategy_part_2 import E_adjusted, months_array, DDt
E1 = E_adjusted
months = months_array
DDt1 = DDt

from Final_project_code_large_stock_part_2 import E_adjusted, DDt
E2 = E_adjusted
DDt2 = DDt

sys.stdout = sys.__stdout__

# Convert months to datetime and sort them
months = pd.to_datetime(months, format='%Y-%m')
sorted_months, sorted_DDt1, sorted_DDt2, sorted_E1, sorted_E2 = zip(*sorted(zip(months, DDt1, DDt2, E1, E2)))

# Subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# DDt subplot
axs[0].plot(sorted_months, sorted_DDt1, label='Strategy Max Drawdown', marker='o')
axs[0].plot(sorted_months, sorted_DDt2, label='200 Max Drawdown', marker='x')
axs[0].set_ylabel("Drawdown (%)")
axs[0].set_title("Max Drawdown Comparison Over Time")
axs[0].set_xlabel("Month")
axs[0].legend()
axs[0].grid(True)
axs[0].tick_params(axis='x', rotation=45)

# Monthly returns
axs[1].plot(sorted_months, sorted_E1, label='Strategy Excess Return', marker='o')
axs[1].plot(sorted_months, sorted_E2, label='200 Excess Return', marker='x')
axs[1].set_xlabel("Month")
axs[1].set_ylabel("Monthly Excess Return")
axs[1].set_title("Monthly Excess Returns Comparison")
axs[1].legend()
axs[1].grid(True)
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
