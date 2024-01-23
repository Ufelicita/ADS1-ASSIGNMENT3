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
    print(clean_df.isnull().sum())

    # Delete first 2 rows
    clean_df = clean_df.iloc[2:112, :]

    # set 1st row as headers # Make the first row as header
    clean_df.columns = clean_df.iloc[0].values.tolist()

    # Delete 1st 2 rows
    clean_df = clean_df.iloc[2:, :]

    # Rename columns
    clean_df = clean_df.rename(columns={
        "Trade (% of GDP)": "Trade",
        "Exports of goods and services (% of GDP)":
        "Exports of goods and services",
        "Foreign direct investment, net inflows (% of GDP)":
        "Foreign direct investment"})

    cols = clean_df.columns[:]
    clean_df = clean_df[cols].apply(pd.to_numeric)

    # check type
    print(clean_df.dtypes)

    return clean_df


# Analyse cleaned data
clean_df = read_transpose_clean(
    "6169b9ce-b902-440b-b18f-78cb7c0bf598_Data.csv")

# Summary statistics for numerical data
print("\nNumerical Summary:")
print(clean_df.describe())


def create_and_save_heatmap(dataframe):
    """
    Group similar columns together, calculate the correlation matrix,
    and create a heatmap.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame containing numeric data.

    Returns:
    - None

    Saves a heatmap plot as 'heatmap.png' and displays the plot.
    """

    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Create a heatmap
    plt.figure(figsize=[10, 8])
    plt.imshow(corr_matrix, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(ticks=range(len(corr_matrix)), labels=corr_matrix.columns,
               rotation=45)
    plt.yticks(ticks=range(len(corr_matrix)), labels=corr_matrix.index)
    plt.title('Correlation Heatmap')

    # Save and show the plot
    plt.savefig("heatmap.png")
    plt.show()

    return


 # Group similar columns together
grouped_df = clean_df.groupby(lambda x: x.split(' ')[0], axis=1).sum()
create_and_save_heatmap(grouped_df)


def plot_scatter_plot(data, x_column, y_column, x_label, y_label,
                      title=None, figsize=(8, 8)):
    """
    Generate a scatter plot.

    Parameters:
    - data: DataFrame containing the data.
    - x_column: Name of the column for the x-axis.
    - y_column: Name of the column for the y-axis.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot (default is None).
    - figsize: Tuple specifying the size of the figure (default is (8, 8)).

    Returns:
    None
    """

    plt.figure(figsize=figsize)
    plt.scatter(data[x_column], data[y_column])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("figure_name.png", dpi=300)
    plt.show()

    return


grouped_df = clean_df.groupby(lambda x: x.split(' ')[0], axis=1).sum()

plot_scatter_plot(grouped_df, 'Trade', 'Foreign',
                  'Trade (% of GDP)', 'Foreign Investments',
                  'A comparison of Trade (% of GDP) and Foreign Investments',
                  figsize=(8, 8))
grouped_df

# set up   scaler object
scaler = pp.RobustScaler()

# extract relevant columns
df_extract = grouped_df.iloc[:, 1:]

# and set up the scaler
scaler.fit(df_extract)

# apply the scaling
norm = scaler.transform(df_extract)

# the results is now a numpy array
print(norm)

# Plot Figure
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1], 10, marker="o")
plt.xlabel("Trade(% of GDP")
plt.ylabel("Foreign Invesements ")
plt.title('Scaled:A comparison of Trade (% of GDP) and Foreign Investments')
plt.show()


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
    silhouette_score = metrics.silhouette_score(data, labels)

    return silhouette_score


def kmeans_clustering(data, n_clusters, indicator1, indicator2, figsize=(8, 8),
                      ):
    """
    Perform K-Means clustering on the input data.

    Parameters:
    - data: DataFrame containing the features for clustering.
    - n_clusters: Number of clusters to form.
    - figsize: Tuple specifying the size of the figure (default is (8, 8)).
    - cmap_name: Name of the colormap for cluster visualization
    (default is "Paired").

    Returns:
    - None
    """
    # Select features for clustering
    trade_norm =  data.loc[:, [indicator1, indicator2]]

    # Standardize the data
    scaler = pp.RobustScaler()
    trade_norm = scaler.fit_transform(trade_norm)

    # Set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    # Fit the data
    kmeans.fit(trade_norm)

    # Get cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and reverse to orginal scales
    centers = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers)

    # Extract x and y values
    xmeans = centers[:, 0]
    ymeans = centers[:, 1]

    # Plot the figure
    plt.figure(figsize=figsize)

    # Plot data with K-Means cluster numbers
    cmap = "paired"
    scatter = plt.scatter(
        data["indicator1"], data["indicator2"], s=50, c=labels, cmap=cmap,
        alpha=0.7)

    # Plot K-Means cluster centers
    plt.scatter(xmeans, ymeans,
                s=100, c="k", marker="D", label='Cluster Centers')

    plt.xlabel("indicator1")
    plt.ylabel("indicator2")

    # Show cluster labels as text annotations
    for i, txt in enumerate(labels):
        plt.annotate(txt, (data[indicator1].iloc[i], 
                           data[indicator2].iloc[i]),
                     fontsize=10, color='red')


    # Show the colorbar
    plt.colorbar(scatter, label='Cluster Labels')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()

    # Plot labels for better visibility
    plt.figure(figsize=figsize)
    plt.scatter(data["indicator1"], data["indicator2"],
                s=50, c=labels, cmap=cmap, alpha=0.7)
    plt.scatter(xmeans, ymeans,
                s=100, c="k", marker="D", label='Cluster Centers')
    plt.xlabel("indicator1")
    plt.ylabel("indicator2")
    plt.title(
        "Global Investment Impact: Unraveling Trade Dynamics through Clustering")

    plt.savefig("figure_name.png", dpi=300)

    plt.show()

    # Add Cluster membership
    data[labels] = labels

    # write to excel file
    data.to_excel("cluster_results.xlsx")
    print(data.head())
    return


# Call the function with cluster number
kmeans_clustering(df_extract, n_clusters=3,
                  indicator1="Foreign", indicator2="Trade")
