# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:55:37 2024

@author: fa22aep
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:31:43 2024

@author: -
"""
# Set OMP_NUM_THREADS to 1
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
from sklearn.preprocessing import RobustScaler
import scipy.optimize as opt
import errors as err


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

    clean_df.index.name = 'Year'

    # Delete first 2 rows
    clean_df = clean_df.iloc[4:67, :]

    # check the dataframe types
    print(clean_df.dtypes)

    # Make data values numeric
    country_columns = clean_df.columns[0:]

    clean_df[country_columns] = clean_df[
        country_columns].apply(pd.to_numeric, errors='coerce')

    # check the dataframe types again
    print(clean_df.dtypes)

    # reset index
    clean_df = clean_df.reset_index()

    return clean_df


# Analyse cleaned data
clean_df = read_transpose_clean(
    "API_SP.DYN.LE00.IN_DS2_en_excel_v2_6508273.csv")

# check the statistics of data
print(clean_df.describe())


def plot_average_life_expectancy(dataframe):
    """
    Plot the average life expectancy for each country.

    Args:
    - dataframe: pandas DataFrame with 'Year' as one of the 
    columns and columns for countries.

    """

    if 'Year' in dataframe.columns:
        dataframe.set_index('Year', inplace=True)

    # Calculate the average life expectancy for each country
    average_life_expectancy = dataframe.mean(axis=0)

    # Create a horizontal bar plot
    plt.figure(figsize=(12, 8))
    average_life_expectancy.sort_values().plot(kind='barh', color='skyblue')

    # Add labels and title
    plt.title('Average Life Expectancy by Country')
    plt.xlabel('Average Life Expectancy')
    plt.ylabel('Country')

    plt.savefig("figure_name.png", dpi=300)
    # Show the plot
    plt.show()

    return


plot_average_life_expectancy(clean_df)

# Transpose dataframe
clean_df_t = clean_df.transpose()

# Calculate % change of life expectancy in 50 years.
value_diff = clean_df_t["2021"] - clean_df_t["1971"]

# Calculate the rate over 50 years for each country
clean_df_t["% change"] = 100.0 / 50.0 * value_diff / clean_df_t["1971"]

# Display the updated DataFrame
print(clean_df_t)


def plot_scatter(df, x_column, y_column, cmap='Paired', size=100):
    """
    Plot a scatter plot to  visually examining the data of  life expectancy 
    in 1971 and its correlation with the percentage change over a 
    50-year period

    Args:
    - df: DataFrame
    - x_column: Column for x-axis
    - y_column: Column for y-axis
    - cmap: Colormap (default: 'Paired')
    - size: Size of the points (default: 100)
    """

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df[x_column], df[y_column], cmap=cmap, s=size)
    plt.title(
        f'Life Expectancy Evolution (1971-2021): {x_column} vs. {y_column}',
        fontsize=18)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return


# Using the function to plot without colorbar
plot_scatter(clean_df_t, "1971", "% change", cmap='Paired', size=100)


def plot_normalized_life_expectancy_trends(df, x_column, y_column, size=100):
    """
    Plot a scatter plot with normalized data.

    Args:
    - df: DataFrame with the index containing country labels
    - x_column: Column for x-axis
    - y_column: Column for y-axis
    - size: Size of the points (default: 100)
    """

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(
        scaler.fit_transform(df[[x_column, y_column]]),
        columns=[x_column, y_column]
    )

    # Plot the scatter plot with normalized data
    plt.figure(figsize=(12, 8))
    plt.scatter(
        normalized_df[x_column], normalized_df[y_column], c='blue', s=size
    )
    plt.title(
        f'Normalized Life Expectancy Evolution vs {y_column}', fontsize=18)

    plt.xlabel(f'Normalized {x_column}', fontsize=14)
    plt.ylabel(f'Normalized {y_column}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

    return


# Using the function to plot with normalized data
plot_normalized_life_expectancy_trends(clean_df_t, "1971",
                                       "% change", size=100)

os.environ["OMP_NUM_THREADS"] = "1"


def compute_silhouette_score(data, n_clusters):
    """
    Calculates silhouette score for n clusters.

   Args:

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
    score = compute_silhouette_score(clean_df_t, ic)
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
    labels:cluster labels 
    """

    # Scale the data using Robustscaler
    scaler = pp.RobustScaler()
    norm = scaler.fit_transform(data)

    # Perform K-Means clustering on the scaled data
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    print(cen)

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
    plt.title("Life Expectancy and its Rate of Change in 1971")

    # Show the colorbar
    plt.colorbar(scatter, label='Cluster Labels')

    # Show the legend
    plt.legend()

    plt.savefig("figure_name.png", dpi=300)

    # Show the plot
    plt.show()

    return labels


def cluster_life_expectancy(df):
    """
    Cluster life expectancy patterns based on 1971 values
    and percentage change.

    Parameters:
    - df: DataFrame with '1971' and '% change' columns.

    Returns:
    - df: DataFrame with an additional 'labels' column 
    indicating cluster membership.
    """

    # Extract 1971 for clustering
    life_expectancy_change = clean_df_t[["1971", "% change"]].copy()

    # Create a scaler object
    scaler = RobustScaler()

    # Set up the scaler
    scaler.fit(life_expectancy_change)

    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=3, n_init=20)

    # Transform data using the scaler
    norm = scaler.transform(life_expectancy_change)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    (print(cen))

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))

    # Plot data with kmeans cluster number
    plt.scatter(life_expectancy_change["1971"],
                life_expectancy_change["% change"],
                50, labels, marker="o", cmap="Paired")

    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 60, "k", marker="d")
    plt.title("Clustered Scatter Plot of Life Expectancy Patterns (1971-2021)")
    plt.xlabel("Life Expectancy in 1971")
    plt.ylabel("Percentage Change in Life Expectancy")
    plt.savefig("clustered_scatter_plot.png", dpi=300)
    plt.show()

    # Add the cluster membership information to the dataframe
    life_expectancy_change["labels"] = labels

    # Write into a file
    life_expectancy_change.to_excel("cluster_results.xlsx")

    return life_expectancy_change


result_df = cluster_life_expectancy("life_expectancy_change")


# Normalize the data
scaler = MinMaxScaler()
result_df[['1971', '% change']] = scaler.fit_transform(
    result_df[['1971', '% change']])

# Group the data by 'labels'
grouped_data = result_df.groupby('labels')

# Define bar width and positions
bar_width = 0.35
labels = sorted(result_df['labels'].unique())

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(
    labels), figsize=(15, 5), sharey=True)

# Custom color palette
# Coral for life expectancy, Burgundy for percentage change
colors = ['#FFA07A', '#800020']

for label, ax in zip(labels, axes):
    data_label = grouped_data.get_group(label)
    countries = data_label.index
    life_expectancy = data_label['1971']
    percentage_change = data_label['% change']

    r1 = np.arange(len(countries))

    # Create grouped bar chart with custom colors
    bars_life = ax.bar(
        r1, life_expectancy, color=colors[0],
        width=bar_width, edgecolor='grey', label='Life Expectancy')
    bars_percentage = ax.bar(r1, percentage_change, bottom=life_expectancy,
                             color=colors[1], width=bar_width,
                             edgecolor='grey', label='% Change')

    # Set labels and title
    ax.set_xlabel('Countries')
    ax.set_title(f'Label {label}')

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, rotation=45, ha='right')

# Set common y-axis label
fig.text(0, 0.5, 'Normalized Values', va='center', rotation='vertical')

# Create a single legend for the entire plot
fig.legend([bars_life, bars_percentage], [
           'Life Expectancy', '% Change'], loc='upper right')

# Set the overall title
fig.suptitle(
    'Comparison of Cluster Labels: Life Expectancy and % Change (1971-2021)',
    fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()
