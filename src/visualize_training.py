import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# ----------------------------------------------- Default Arguments ----------------------------------------------------

batch_size = 1
PATH = 'model.csv'


# ----------------------------------------------- Parsed Arguments -----------------------------------------------------

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--path", help="Set path to model file.")
parser.add_argument("--vis_type", help="Set visualization type.")

# Read arguments from the command line
args = parser.parse_args()

# Check arguments
print(103*"-")
if args.path:
    PATH = args.path
out = "| PATH: " + PATH
print(out, (100 - len(out))*' ', '|')
print(103*"-")


# ----------------------------------------------- Visualization --------------------------------------------------------
# Read data frame on csv file
df = pd.read_csv(PATH)

fig = 1
# Plot the loss throughout epochs
plt.figure(fig)
sns.set_theme(style="darkgrid")
sns.lineplot(x="epoch", y="loss_avg", data=df)
sns.lineplot(x="epoch", y="loss_avg", data=df)
plt.show()
