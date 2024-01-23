# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:31:43 2024

@author: -
"""

import os
import pandas as pd
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.datasets as skdat
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
# Set OMP_NUM_THREADS to 1


def read_transpose_clean(file):
    """
    Reads, transposes, and cleans the data.
    """

    # Read the data
    df = pd.read_csv(file)

    # Set the display format for float values
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Transpose dataframe
    df_t = pd.DataFrame.transpose(df)

    # Drop NANs
    clean_df = df_t.dropna()

    # check for NANs
    print(clean_df.isna().any())

    # set 1st row as headers # Make the first row as header
    clean_df.columns = clean_df.iloc[0].values.tolist()

    # Delete first 2 rows
    clean_df = clean_df.iloc[4:67, :]

    # reset index
    clean_df = clean_df.reset_index()

    return clean_df


# Analyse cleaned data
clean_df = read_transpose_clean(
    "API_SP.DYN.LE00.IN_DS2_en_excel_v2_6508273.csv")

# Creating a subset for use for Clustering
life_df = clean_df.transpose()

print(life_df.describe())

life_df = life_df.reset_index()

# set 1st row as headers # Make the first row as header
life_df.columns = life_df.iloc[0].values.tolist()

# remove 1st row
life_df = life_df.iloc[1:35]

year_columns = life_df.columns[1:]

# Convert the selected columns to numeric
life_df[year_columns] = life_df[year_columns].apply(pd.to_numeric,
                                                    errors='coerce')

# Display the data types of the columns
print(life_df.dtypes)

print(clean_df.isna().any())

# creating a dataframe with just 1971 and 2021
life_exp = life_df[['index', '1971', '2021']].copy()

life_exp = life_exp.set_index(["index"])

value_diff = life_exp['2021'] - life_exp['1971']

# calculate the growth over 40 years
life_exp["Change Rate"] = 100.0/40.0 * value_diff / life_exp["1971"]


print(life_exp.describe())

# Create a scatter plot to visualize life expectancy trends
plt.figure(figsize=(8, 8))
plt.scatter(life_exp["1971"], life_exp["Change Rate"])
plt.xlabel("Life Expectancy in 1971")
plt.ylabel("Change Rate [%]")
plt.title("Life Expectancy Trends Over Time")
plt.show()

columns = ["1971", "Change Rate"]
life_exp_scaled = life_exp[columns]


def plot_normalized_life_expectancy_trends(data):
    """
    Plot life expectancy data using MinMaxScaler.

    Args:
    - life_exp: DataFrame containing life expectancy data.
    - columns: List of columns to plot (default is ["1971", "Change Rate"]).

    Returns:
    - None
    """

    # Create a scaler object
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(life_exp_scaled)

    # Apply the scaling
    life_exp_norm = scaler.transform(life_exp_scaled)

    # Plot the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(life_exp_norm[:, 0], life_exp_norm[:, 1])
    plt.xlabel(f"{columns[0]} (Normalized)")
    plt.ylabel(f"{columns[1]} (Normalized)")
    plt.title("Scaled Life Expectancy Trends")
    plt.show()
    return


plot_normalized_life_expectancy_trends(life_exp)


os.environ["OMP_NUM_THREADS"] = "1"


def calculate_silhouette_score(data, n_clusters):
    """
    Calculates silhouette score for n clusters.

    Parameters:

    - data: The input data is a pandas DataFrame).
    - n_clusters: The number of clusters.

    Returns:
    - Silhouette score.
    """


    # Initialize KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(data)

    # Get cluster labels
    labels = kmeans.labels_

    # Calculate the silhouette score
    silhouette_score = score = (skmet.silhouette_score(data, labels))

    return silhouette_score


# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = calculate_silhouette_score(life_exp_scaled, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")


def kmeans_clustering_visualization(data, feature_indices,
                                    n_clusters=3, figsize=(8, 8),
                                    cmap_name="Paired"):
    """
    Perform K-Means clustering on the input data and visualize the results.

    Args:
    - data: DataFrame containing the features for clustering.
    - feature_indices: List of column indices for clustering.
    - n_clusters: Number of clusters to form
    - figsize: Tuple specifying the size of the figure (default is (8, 8)).
    - cmap_name: Name of the colormap for cluster visualization

    Returns:
    - None
    """


    # Scale the data using StandardScaler
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(data)

    # Perform K-Means clustering on the scaled data
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)

    # Plot the figure
    plt.figure(figsize=figsize)

    # Plot data with K-Means cluster number
    cmap = cm.get_cmap(cmap_name)
    scatter = plt.scatter(data.iloc[:, feature_indices[0]],
                          data.iloc[:, feature_indices[1]], s=50,
                          c=labels, cmap=cmap, alpha=0.7)

    # Show K-Means cluster centers
    plt.scatter(cen[:, 0], cen[:, 1], s=100,
                c="k", marker="D", label='Cluster Centers')

    plt.xlabel("Life expectancy 1971")
    plt.ylabel("Life Expectancy Rate(%)")

    # Show the colorbar
    plt.colorbar(scatter, label='Cluster Labels')

    # Show the legend
    plt.legend()

    plt.savefig("figure_name.png", dpi=300)

    # Show the plot
    plt.show()
        
    return

# Example usage
kmeans_clustering_visualization(life_exp,
                                feature_indices=[0, 1],
                                n_clusters=3)

