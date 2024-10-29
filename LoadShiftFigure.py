from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib

def load_shifting_allgemein():
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

    plt.plot(time, consumer_load, label='$P_{consumer}$', color="blue")
    plt.plot(time, prosumager_load, label='$P_{prosumager}$', color="black", linewidth=2)
    plt.fill_between(time, consumer_load, prosumager_load, where=(consumer_load > prosumager_load),
                    color='forestgreen', interpolate=True, label='$E_{shifted}$')
    plt.fill_between(time, consumer_load, prosumager_load, where=(consumer_load <= prosumager_load),
                    color='crimson', interpolate=True, label='$E_{increased}$')

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
    plt.savefig(Path(__file__).parent / "LoadShifting.svg")
    plt.show()


def load_shifting_thermal_loss():
    # show how the thermal loss of the thermal mass is calculated by first pre heating and then de-loading
    indoor_temp = [20, 20, 22, 22, 22] + list(np.geomspace(2, 0.01, num=5)+20) + [20 for _ in range(14)]
    log_mass = list(np.geomspace(0.01, 1, num=15)+18)
    log_mass.sort(reverse=True)
    log_mass = log_mass + [18 for _ in range(4)]
    x_axis= np.arange(1, 25)
    thermal_mass_temp = [18, 18, 18.4, 18.7, 19] + log_mass
    
    log_power = [14-x for x in list(np.geomspace(0.01, 4, num=15) + 4)] + [10 for _ in range(3)]
    log_power.sort(reverse=False)
    q_charging= [10, 10, 18, 17, 16, 15] + log_power
    # q_discharging = [10 for _ in range(4)] + log_power
    
    matplotlib.rc("font", **{"size": 18})
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(x_axis, indoor_temp, label="indoor temperature", color="orange")
    ax.plot(x_axis, thermal_mass_temp, label="thermal mass temperature", color="blue")
    ax.plot(x_axis, np.full(24, fill_value=18), label="steady thermal mass temperature", linestyle=":", color="blue")
    ax.plot(x_axis, np.full(24, 20), label="steady room temperature", linestyle=":", color="orange")

    ax2 = ax.twinx()
    ax2.plot(x_axis, q_charging, label="heating power", color="firebrick", )
    ax2.fill_between(x_axis, q_charging, np.full(24, 10), where=q_charging>=np.full(24, 10), color="red", alpha=0.3, interpolate=True)
    ax2.fill_between(x_axis, q_charging, np.full(24, 10), where=q_charging<=np.full(24, 10), color="green", alpha=0.3, interpolate=True)

    ax2.plot(x_axis, np.full(24, 10), label="constant heating power", linestyle=":", color="firebrick")

    ax.legend(handles=[
            Line2D([0], [0], color="firebrick", linewidth=1.5, ),
            Patch(edgecolor="white", facecolor="red", alpha=0.3, ),
            Patch(edgecolor="white", facecolor="green"),
            Line2D([0], [0], color="orange", linewidth=1.5, ),
            Line2D([0], [0], color="blue", linewidth=1.5, ),
            Line2D([0], [0], color="white", linewidth=1.5,),
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle=":")
        ],
        labels=[
            "heating power (kW)", "additional heating energy (kWh)", "reduced heating energy (kWh)", "indoor temperature (°C)", "thermal mass temperature (°C)", "", "shifting", "steady state"
        ]
    )
    ax.set_xlabel("hours")
    ax.set_ylabel("temperature in °C")
    ax2.set_ylabel("heating power in kW", color="firebrick")
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.spines['right'].set_color('firebrick')

    plt.savefig(Path(__file__).parent / "Diss Graphiken" / f"Load_shifting_temperatures_and_power.png")
    plt.savefig(Path(__file__).parent / "Diss Graphiken" / f"Load_shifting_temperatures_and_power.svg")


if __name__ == "__main__":
    load_shifting_thermal_loss()

