import numpy as np
import matplotlib.pyplot as plt


consumer_load = np.array([
    3,
    3,
    2,
    2,
    4,
    3,
    5,
    8,
    7,
    5,
    4,
    5,
    5,
    6,
    7,
    7,
    9,
    11,
    12,
    10,
    8,
    7,
    4,
    3,
])
prosumager_load = np.array([
    3,
    3,
    2,
    2,
    7,  # + 3
    4,  # +1
    4,  # -1
    6,  # -2
    6,  # -1
    5,
    4,
    5,
    5,
    6,
    10,  # +3
    8,  # +1
    9,
    9,  # -2
    10,  # -2
    10,
    8,
    7,
    4,
    3,
])
price = np.array([
                  2,
                  2,
                  2,
                  2,
                  2,
                  2,
                  5,
                  5,
                  5,
                  3,
                  3,
                  3,
                  3,
                  3,
                  3,
                  3,
                  6,
                  7,
                  7,
                  6,
                  5,
                  5,
                  4,
                  2,
                  ])

time = np.arange(1, 25)

plt.plot(time, consumer_load, label='Consumer Load', color="blue")
plt.plot(time, prosumager_load, label='Prosumager Load', color="black", linewidth=2)
plt.fill_between(time, consumer_load, prosumager_load, where=(consumer_load > prosumager_load),
                 color='forestgreen', interpolate=True, label='reduced peak')
plt.fill_between(time, consumer_load, prosumager_load, where=(consumer_load <= prosumager_load),
                 color='crimson', interpolate=True, label='higher peak')

plt.legend(loc='upper left')
plt.xticks([])  # Hide x-axis ticks
plt.yticks([])  # Hide primary y-axis ticks
# Create a secondary y-axis for price
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(time, price, color='magenta', linestyle='--', label='Price')
ax2.set_ylabel('Price')
ax2.legend(loc='upper right')
ax2.yaxis.set_ticks([])  # Hide secondary y-axis ticks
# Set primary y-axis label

ax.set_ylabel('Load')
ax.set_xlabel('Time')
plt.show()
