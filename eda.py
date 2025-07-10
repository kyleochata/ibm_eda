def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import os
import numpy as np
import pandas as pd
import requests
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

os.makedirs("data", exist_ok=True)

# URL and file path
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv"
file_path = "data/iris_data.csv"

download(url, file_path)

data = pd.read_csv(file_path)
print(data.head())

# Q1: Determine the following:

# The number of data points (rows). (Hint: check out the dataframe .shape attribute.)
# The column names. (Hint: check out the dataframe .columns attribute.)
# The data types for each column. (Hint: check out the dataframe .dtypes attribute.)

#.shape will return (rows, rols)
rows, cols = data.shape
print(rows)

# .columns to get col names as array; tolist() to separate it out of the tuple returned from .columns
columns = data.columns.tolist()
print(columns)

print(data.dtypes)

# Examine the species names and note that they all begin with 'Iris-'. Remove this portion of the name so the species name is shorter.
# Hint: there are multiple ways to do this, but you could use either the string processing methods or the apply method.
# str method maps the following function to each entry as a string
data['species'] = data.species.str.replace('Iris-', '')
# Alternative
# data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))
print(data.head())

# Determine the following:
# The number of each species present. (Hint: check out the series .value_counts method.)
# The mean, median, and quantiles and ranges (max-min) for each petal and sepal measurement.
species_count = data["species"].value_counts()
print(species_count)

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
# for col in num_cols:
#     print(f"\n Stats for {col}:")
#     print(f"Mean: {data[col].mean()}")
#     print(f"Median: {data[col].median()}")
#     print(f"Max: {data[col].max()}")
#     print(f"Min: {data[col].min()}")
stats = {}
for col in num_cols:
    stats[col] = {
        'mean': float(data[col].mean()),
        'median': float(data[col].median()),
        'max': float(data[col].max()),
        'min': float(data[col].min()),
        'range': float(data[col].max() - data[col].min())
    }

for key, val in stats.items():
    print(f"{key}: {val}")

stats_described = data.describe()
print(stats_described)

# alternative: select desired rows from .describe and add in 'median'
stats_df = data.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
out_fields = ['mean', '25%', '50%', '75%', 'range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%': 'median'}, inplace=True)
print(stats_df)

# calc mean & median for each measurement by species in a separate dataframe

# mean
sp_means = data.groupby('species').mean()
print(f"Species mean: {sp_means}")

sp_median = data.groupby('species').median()
print(f"Species Median: {sp_median}")

# Apply multiple functions at once
sp_mandm = data.groupby("species").agg(['mean', 'median']) # passs list of recognized strings
sp_mandm2 = data.groupby("species").agg([np.mean, np.median]) # pass explicit aggregation function

print(f"list_string: {sp_mandm}")
print(f"explicit agg: {sp_mandm2}")

# When fields need to be aggregated differently:
agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
# agg_dict = {}
# for field in data.columns:
#   if field != 'species':
#       agg_dict[field] = ['mean', 'median']
agg_dict['petal_length'] = 'max'
print('====agg_dict====')
pprint(agg_dict)
g = data.groupby("species").agg(agg_dict)
print(g)

# Scatter plot of sepal_lenght vs sepal_width with Matplotlib
ax = plt.axes()
ax.scatter(data["sepal_length"], data["sepal_width"])
ax.set(xlabel="Sepal Length(cm)",
       ylabel="Sepal Width(cm)",
       title="Sepal Length vs Width"
       )
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/Sepal_length_vs_width_scatter.png', dpi=300, bbox_inches="tight")
plt.close()

# histogram - distribution of features
hist = plt.axes()
hist.hist(data['petal_length'], bins=25)
hist.set(xlabel="Petal Length (cm)",
         ylabel= "Freqeuncy",
         title="Distribution of Petal Lengths"
         )
plt.savefig('plots/dist_petal_lengths.png', dpi=300, bbox_inches="tight")
plt.close()

# single plot with histograms for ea feature overlayed vs separate histograms in a grid
# Set style
sns.set_context('notebook')

# Option 1: Overlaid histograms (all in one plot)
plt.figure()
overlay_hist = data.plot.hist(bins=25, alpha=0.5)
overlay_hist.set_xlabel('Size (cm)')
plt.savefig('plots/histOverlay.png', dpi=300, bbox_inches="tight")
plt.close()
# Option 2: Separate histograms in grid 
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 features
data.hist(bins=25, ax=axs)

# Set labels for bottom row
for ax in axs[-1, :]:  # Last row
    ax.set_xlabel('Size (cm)')

# Set labels for first column
for ax in axs[:, 0]:  # First column
    ax.set_ylabel('Frequency')
plt.savefig('plots/ft_histo_combined.png', dpi=300, bbox_inches="tight")
plt.close()

# boxplot of each petal and sepal measurment
data.boxplot(by="species")
plt.savefig('plots/species_boxplot.png', dpi=300, bbox_inches="tight")
plt.close()

print("groupby test ======")
pprint(data.groupby("species")["petal_length"].describe())