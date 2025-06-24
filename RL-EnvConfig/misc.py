import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Data/rx_tx_data_digitalLNA_1.xlsx")

df.info()

df.head()

# # Load both CSVs
# df1 = pd.read_csv("dqn_snr_improvement_log.csv")  # e.g., from model A
# df2 = pd.read_csv("sac_snr_improvement_log.csv")  # e.g., from model B

# # Ensure time step is aligned (optional but recommended)
# if "Time Step" not in df1.columns:
#     df1["Time Step"] = range(len(df1))
# if "Time Step" not in df2.columns:
#     df2["Time Step"] = range(len(df2))

# # Plot SNR Improvement from both
# plt.figure(figsize=(10, 5))
# plt.plot(df1["Time Step"], df1["SNR Improvement"], label="DQN", color = 'orange')
# plt.plot(df2["Time Step"], df2["SNR Improvement"], label="SAC", color = 'blue')
# plt.axhline(y=0, color='gray', linestyle='--')

# plt.title("SNR Improvement: DQN vs. SAC")
# plt.xlabel("Time Step")
# plt.ylabel("SNR Improvement (dB)")
# plt.legend()
# plt.tight_layout()
# plt.show()