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

    # Create figure/axes for finer control
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main series with clearer distinction
    ax.plot(
        time,
        consumer_load,
        label='$P_{simulation}$',
        color="#1f77b4",  # tab:blue
        linewidth=2.0,
        marker='o',
        markersize=4,
        markerfacecolor='white',
        markeredgewidth=1,
        markevery=3,
        zorder=3,
    )
    ax.plot(
        time,
        prosumager_load,
        label='$P_{optimization}$',
        color="#222222",
        linewidth=2.8,
        marker='s',
        markersize=4,
        markerfacecolor='white',
        markeredgewidth=1,
        markevery=3,
        zorder=4,
    )

    # Pleasant green/red for shifted/increased energy with gentle transparency
    shifted_color = "#4CAF50"   # green
    increased_color = "#E53935"  # red

    f1 = ax.fill_between(
        time,
        consumer_load,
        prosumager_load,
        where=(consumer_load > prosumager_load),
        color=shifted_color,
        alpha=0.3,
        interpolate=True,
        label='$E_{shifted}$',
        zorder=2,
    )
    f2 = ax.fill_between(
        time,
        consumer_load,
        prosumager_load,
        where=(consumer_load <= prosumager_load),
        color=increased_color,
        alpha=0.3,
        interpolate=True,
        label='$E_{increased}$',
        zorder=1,
    )

    # Hide ticks as before for a clean look
    ax.set_xticks([])
    ax.set_yticks([])

    # Secondary y-axis for price with a calmer accent color
    ax2 = ax.twinx()
    price_line = ax2.plot(
        time,
        price,
        color='tab:orange',
        linestyle='--',
        linewidth=1.8,
        label='Price',
        zorder=3,
    )[0]
    ax2.set_ylabel('Price', color='tab:orange', fontsize=16)
    ax2.tick_params(axis='y', colors='tab:orange')
    ax2.spines['right'].set_color('tab:orange')
    ax2.set_yticks([])

    # Labels
    ax.set_ylabel('Load', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)

    # Single combined legend to avoid duplication/clutter
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax.legend(all_handles, all_labels, loc='upper left', frameon=False, ncol=1, fontsize=13)

    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "LoadShifting.svg")
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
    load_shifting_allgemein()

