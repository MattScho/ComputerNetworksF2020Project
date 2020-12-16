import pandas as pd
import matplotlib.pyplot as plt
import glob as glob

'''
Settings
'''
# Directory containing results
results_directory = "../results/b/"

# Metric improvement to visualize and y limit range
# Either ["Reduced Unservice", "Service Level", "Budget"]
metric = "Service Level"

# Vary this to focus on a particular y range, though pyplot's zoom feature could be used
# If interested in viewing the Budget metric comment out line "plt.setp(axs, ylim=ylimits)"
ylimits = [0.5, 1.0]

'''
Processing
'''
# Step through budgets
for budget in [500]:
    # Budget file name pattern
    name_pattern = "*steps*" + str(budget) + ".csv"

    # Learning rate files
    step_files = glob.glob(results_directory + name_pattern)

    # Construct 2x2 plot
    fig, axs = plt.subplots(2, 3)
    plt.setp(axs, ylim=ylimits)

    # Step through retrieved files
    for i, file in enumerate(step_files):
        # Read in file
        frame = pd.read_csv(file, names=["Reduced Unservice", "Service Level", "Budget"])

        # Plot on axs
        axs[int(i/3), i%3].scatter(frame.index.values, frame[metric].values, s=3)

        # Grab version to make title
        v = file.split("v")[-2][0]
        axs[int(i/3), i%3].set_title("Methodology " + v)

    fig.suptitle("Learning Rate - Budget " + str(budget) + "\nUniform Arrival Uniform Destinations")

    # Show 2x2 plot
    plt.show()


