import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

soc0 = 0
max_capcity = 100
max_charge = 10
max_discharge = -5
discharge = -3  # pro stunde
time = np.arange(24)

# prices = [np.sin(2*np.pi * np.linspace(0, 24) / 24) + 1 + np.random.normal(0, 0.1) for _ in range(10)]
np.random.seed(10)
price = np.random.rand(24) 
num_scenarios = 10
price_std_dev = 0.05
price_scenarios = np.random.normal(loc=price, scale=price_std_dev, size=(num_scenarios, 24))
probabilities = np.full(shape=(1, num_scenarios), fill_value=1/num_scenarios)


# Todo include stochastic
charge = cp.Variable((1,24), "charging_power")
state_of_charge = cp.Variable((1,24), name="battery_capacity", nonneg=True)

p = cp.Parameter((price_scenarios.shape), name="price", value=price_scenarios)
probability = cp.Parameter((1, price_scenarios.shape[0]), name="price probability", value=probabilities)

constraints = []
constraints.append(state_of_charge[:, 0] == soc0)
# for t in time:
#     if t in np.arange(7,18):
#         constraints.append(capacity[t] == capacity[t-1] + discharge)
#         constraints.append(charge[t]==0)
#     else:
#         constraints.append(capacity[t] == capacity[t-1] + charge[t])

#     constraints.append(capacity[t] <= max_capcity)
#     constraints.append(capacity[t] >= 0)

constraints.append(state_of_charge[:, 1:7] == state_of_charge[:, :6] + charge[:, :6])

constraints.append(charge[:, 7:17]==0)
constraints.append(state_of_charge[:, 7:18] == state_of_charge[:, 6:17] + discharge)

constraints.append(state_of_charge[:, 18:] == state_of_charge[:, 17:23] + charge[:,17:23])


constraints.append(state_of_charge[:, :] <= max_capcity)
constraints.append(charge <= max_charge)
constraints.append(charge >= max_discharge)

constraints.append(state_of_charge[:, 23] >= 20)


objective = cp.Minimize(cp.sum(probability @ (cp.multiply(p, charge))))

problem = cp.Problem(objective=objective, constraints=constraints)
problem.solve()

if problem.status not in ["infeasible", "unbounded"]:
    # Output the results
    print("Optimal charge values:", charge.value)
    print("Optimal capacity values:", state_of_charge.value)

    # Plot the results
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax2 = plt.twinx()
    ax.plot(time + 1, charge.flatten().value, label="Charging Power")
    ax.plot(time + 1, state_of_charge.flatten().value, label="Battery Capacity")
    for i in range(num_scenarios):

        ax2.plot(time, price_scenarios[i], label="price", color="red", alpha=0.2)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    plt.title("Optimal Charging Schedule and Battery Capacity")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Problem could not be solved. Status:", problem.status)