# %% 
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from run_experiments import RESULTS_DIR

# %% 
df = pd.read_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index_col="idx")

# %% 
# Set the figure size and style
plt.figure(figsize=(8, 6))
sns.set(style='whitegrid')

# Create the line plot
sns.lineplot(data=df, x='n_actual', y='runtime', hue='n_planned', palette=["blue", "red"])

# Set the plot title and axis labels
plt.title('Runtime vs. amount of actual routes')
plt.xlabel('Amount of actual routes')
plt.ylabel('Runtime in seconds')

# Set the legend
plt.legend(title='Amount of planned routes')

# Adjust the plot margins
plt.tight_layout()

# Save the plot as an image (optional)
plt.savefig('line_plot.png', dpi=300)

# Display the plot
plt.show()
