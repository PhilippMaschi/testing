import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path


# Generate dummy data
np.random.seed(42)  # For reproducibility
length = 48  # Length of arrays

# Creating dummy profiles
consumer_profile = np.random.uniform(10, 30, length)
prosumager_profile = consumer_profile + np.random.uniform(-5, 8, length)  # some variation around consumer profile
national_demand_profile = np.random.uniform(200, 300, length)
price_profile = np.random.uniform(10, 100, length)  # random price profile
price_threshold = 50  # arbitrary threshold for "renewable" energy


def flexibility_factor(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> float:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted."""
    difference = consumer_profile - prosumager_profile
    difference[difference<0] = 0  # only consider energy that is shifted "away", so the reduced energy

    return  difference.sum() / national_demand_profile.sum() * 100  # %

def flexibility_factor_hourly(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> np.array:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted in every hour as percentage"""
    difference = consumer_profile - prosumager_profile
    difference[difference<0] = 0  # only consider energy that is shifted "away", so the reduced energy

    return  difference / national_demand_profile * 100  # %

def supply_and_demand_matching(price_profile: np.array, price_threshold: float, consumer_profile: np.array, prosumager_profile: np.array) -> float:
    """set price threshold and if the price is below that threshold the used electricity is “renewable”. Check how much more “renewable” electricity can be used
    returns the differnce betwen prosumager demand at low prices and consumer demand at low prices"""

    consumer_match = 0
    prosumager_match = 0
    for i, price in enumerate(price_profile):
        if price <= price_threshold:
            consumer_match += consumer_profile[i]
            prosumager_match += prosumager_profile[i]

    return prosumager_match - consumer_match    


def flexible_storage_efficiency(consumer_profile: np.array, prosumager_profile: np.array) -> float:
    """calculates the upward storage efficiency between 0 and 1"""
    # diff negative is the energy that is taken out of the building during "discharging"
    discharging_energy = prosumager_profile - consumer_profile
    discharging_energy[discharging_energy > 0] = 0

    # diff positve is the energy that is stored in the building during "charging"
    charging_energy = prosumager_profile - consumer_profile
    charging_energy[charging_energy < 0] = 0

    #  eta from https://doi.org/10.1016/j.apenergy.2017.04.061
    eta = 1 - (prosumager_profile - consumer_profile).sum() / charging_energy.sum()
    # if eta is negative than the prosumager uses less energy! which can be the case if there is PV for example
    return eta


flex_factor = flexibility_factor(consumer_profile, prosumager_profile, national_demand_profile)
sd_matching = supply_and_demand_matching(price_profile, price_threshold, consumer_profile, prosumager_profile)
storage_efficiency = flexible_storage_efficiency(consumer_profile, prosumager_profile)

print(f"Flexiblity Factor: {flex_factor} %")
print(f"Supply and Demand matching: {sd_matching} kWh")
print(f"Storage efficiency: {storage_efficiency*100} %")

plt.figure(figsize=(12, 8))

# Consumer and Prosumager Profile Plot
plt.subplot(3, 1, 1)
plt.plot(consumer_profile, label="Consumer Profile")
plt.plot(prosumager_profile, label="Prosumager Profile")
plt.title("Consumer vs Prosumager Profile")
plt.xlabel("Time (hours)")
plt.ylabel("Energy (kWh)")
plt.legend()

# Price Profile with Threshold
plt.subplot(3, 1, 2)
plt.plot(price_profile, label="Price Profile")
plt.axhline(price_threshold, color="r", linestyle="--", label="Price Threshold")
plt.title("Price Profile with Threshold")
plt.xlabel("Time (hours)")
plt.ylabel("Price (€/kWh)")
plt.legend()

# National Demand Profile
plt.subplot(3, 1, 3)
plt.plot(national_demand_profile, label="National Demand Profile")
plt.title("National Demand Profile")
plt.xlabel("Time (hours)")
plt.ylabel("Energy (kWh)")
plt.legend()

plt.tight_layout()
plt.show()

