import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tot_time = 4
dt = 0.05
# Load the CSV file into a DataFrame
df_TD3 = pd.read_csv('/Users/ayushsharma/One_Drive_IITD/Design_Project/Codes/CSTR/TD3/With Noise/HIRO/CSTR_TD3.csv', header=None)
df_HIRO_TD3 = pd.read_csv('/Users/ayushsharma/One_Drive_IITD/Design_Project/Codes/CSTR/TD3/With Noise/HIRO/CSTR_HIRO.csv', header=None)
df_HIRO_PPO = pd.read_csv('/Users/ayushsharma/One_Drive_IITD/Design_Project/Codes/CSTR/PPO/With Noise/HIRO/HIRO/CSTR_HIRO_PPO.csv', header=None)

row_index_TD3 = 0  # For example, to get the second row we equate to 1 (index starts from 0)
specific_row_TD3 = df_TD3.iloc[row_index_TD3]

print(specific_row_TD3)

row_index_HIRO_TD3 = 1  # For example, to get the second row we equate to 1 (index starts from 0)
specific_row_HIRO_TD3 = df_HIRO_TD3.iloc[row_index_HIRO_TD3]

print(specific_row_HIRO_TD3)

row_index_HIRO_PPO = 298  # For example, to get the second row we equate to 1 (index starts from 0)
specific_row_HIRO_PPO = df_HIRO_PPO.iloc[row_index_HIRO_PPO]

print(specific_row_HIRO_PPO)



plt.figure()
time = np.linspace(0, tot_time, int(tot_time / dt))
# Plot the first data series
plt.plot(time,specific_row_TD3, label='TD3')

# Plot the second data series
plt.plot(time,specific_row_HIRO_TD3, label='HIRO TD3')

# Plot the third data series
plt.plot(time,specific_row_HIRO_PPO, label='HIRO PPO')

T1 = 0.143  # target
ta = np.ones(int(tot_time / dt)) * T1
plt.plot(time, ta, color='tab:orange', linewidth=2, label='reference concentration')
plt.title('Comparison')
plt.xlabel('time (Hours)')
plt.ylabel('Propylene Glycol')
plt.legend()
plt.savefig('Comparision.png', bbox_inches = 'tight')
plt.close()
# Add titles and labels


# Show the plot
plt.show()
