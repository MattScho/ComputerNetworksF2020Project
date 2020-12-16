import pandas as pd
import matplotlib.pyplot as plt
import glob
from statistics import stdev


'''
Settings
'''
# Directory containing results
results_directory = "../results/c/"

# Metric improvement to visualize and y limit range
# Either ["Reduced Unservice", "Service Level", "Budget"]
metric = "Service Level"

# Vary this to focus on a particular y range, though pyplot's zoom feature could be used
# If interested in viewing the Budget metric comment out line "plt.setp(axs, ylim=ylimits)"
ylimits = [0.5, 1.0]

'''
Process
'''
def bar_plot(ax, data, yerr, colors=None, total_width=0.8, single_width=1, legend=True):
    """
    Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, yerr=yerr[name][x], width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    versionMapper = {
        "noAgent": "No Agent",
        "opt": "Empirically Optimal",
        "v1": "v1 - Paper baseline",
        "v2": "v2 - Reduced input",
        "v3": "v3 - Incentive to move to",
        "v4": "v4 - v2 & v3",
        "v5": "v5 - Per User",
        "v6": "v6 - v4 with prediction"
    }
    # Draw legend if we need
    if legend:
        ax.legend(bars, [versionMapper[k] for k in data.keys()])

# Initialize data and error recorders
data = {
    "opt": [],
    "noAgent": [],
    "v1": [],
    "v2": [],
    "v3": [],
    "v4": [],
    "v5": [],
    "v6": []
}

errs = {
    "opt": [],
    "noAgent": [],
    "v1": [],
    "v2": [],
    "v3": [],
    "v4": [],
    "v5": [],
    "v6": []
}

# Set budgets
budgets = [500, 1000, 2000, 10000]

# Step through budgets
for budget in budgets:
    # Budget file name pattern
    name_pattern = "*steps*" + str(budget) + ".csv"

    # Learning rate files
    step_files = glob.glob(results_directory + name_pattern)

    # Add no agent and opt files
    step_files.append(results_directory + "\\opt_steps.csv")
    step_files.append(results_directory + "\\noAgent_steps.csv")

    # Step through retrieved files
    for i, file in enumerate(step_files):
        print(file)

        # Read in file
        frame = pd.read_csv(file, names=["Reduced Unservice", "Service Level", "Budget"])

        # Get results, using the last 100 indices of the steps
        results = frame[metric].values[-100:]

        # Grab version to make title
        v = file.split("_steps")[-2].split("\\")[-1]
        print(v)


        # Store metric calculations
        errs[v].append(stdev(results))
        data[v].append(sum(results) / len(results))

# Construct plot
fig, ax = plt.subplots()
plt.setp(ax, ylim=ylimits)

# Plot data
bar_plot(ax, data, errs, total_width=.8, single_width=.9)

# Style plot axis X ticks
ax.set_xticks(range(len(budgets)))
ax.set_xticklabels([str(b) for b in budgets])

# Set plot labels
plt.title("Service Level per Methodology and Budget\nUniform Arrival Gaussian Destinations")
plt.ylabel("Service Level")
plt.xlabel("Budget ($ - USD)")

# Show plots
plt.show()
