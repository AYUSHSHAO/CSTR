import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tot_time = 15*24*60
dt = 60
# Load the CSV file into a DataFrame
df_TD3 = pd.read_csv('./TD3/Protein_TD3_reward.csv', header=None)
df_HIRO = pd.read_csv('./HIRO/Protein_HIRO_reward.csv', header=None)

row_index_TD3 =30  # For example, to get the second row (index starts from 0)
specific_row_TD3 = df_TD3.iloc[row_index_TD3]

print(specific_row_TD3)

row_index_HIRO = 48  # For example, to get the second row (index starts from 0)
specific_row_HIRO = df_HIRO.iloc[row_index_HIRO]

print(specific_row_HIRO)

plt.figure()
time = np.linspace(0, tot_time, int(tot_time / dt))
# Plot the first data series
plt.plot(time,specific_row_TD3, label='TD3')

# Plot the second data series
plt.plot(time,specific_row_HIRO, label='HIRO')

T1 = 590  # target
ta = np.ones(int(tot_time / dt)) * T1
plt.plot(time, ta, color='tab:orange', linewidth=2, label='reference concentration')
plt.title('Values from Two Different CSV Files')
plt.xlabel('time (min)')
plt.ylabel('Protein')
plt.legend()
plt.savefig('Comparision.png', bbox_inches = 'tight')
plt.close()
# Add titles and labels


# Show the plot
plt.show()
