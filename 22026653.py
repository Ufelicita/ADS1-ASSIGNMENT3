# -*- coding: utf-8 -*-
"""
ADS 1: Clustering and Fitting Assignment 3
Created on Thu Jan 25 20:55:37 2024

@author: fa22aep
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


def Visualise_Life_expectancy_evolution(df, x_column,
                                        y_column, cmap='Paired',
                                        size=100):
    """
    Plot a scatter plot to  visually examine the data of  life expectancy 
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


# Using the function to plot
Visualise_Life_expectancy_evolution(clean_df_t,
                                    "1971", "% change",
                                            cmap='Paired', size=100)


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
    scaler = RobustScaler()
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

    # Plot labels and ticks
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


# calculate silhouette score for 2 to 10 clusters with function
for ic in range(2, 11):
    score = compute_silhouette_score(clean_df_t, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")


def kmeans_clustering_life_expectancy(df):
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
    kmeans = KMeans(n_clusters=3, n_init=20)

    # Transform data using the scaler
    norm = scaler.transform(life_expectancy_change)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))

    # Plot data with kmeans cluster number
    scatter = plt.scatter(life_expectancy_change["1971"],
                          life_expectancy_change["% change"],
                          50, labels, marker="o", cmap="Paired",
                          label="Data Points")

    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 60, "k", marker="d", label="Cluster Centers")

    plt.title("Clustered Scatter Plot of Life Expectancy Patterns (1971-2021)")
    plt.xlabel("Life Expectancy in 1971")
    plt.ylabel("Percentage Change in Life Expectancy")

    # Show the legend with cluster numbers
    legend_labels = [f"Cluster {i}" for i in range(3)]
    plt.legend(handles=scatter.legend_elements()[0], title="Legend",
               labels=legend_labels)

    plt.savefig("clustered_scatter_plot.png", dpi=300)
    plt.show()

    # Add the cluster membership information to the dataframe
    life_expectancy_change["labels"] = labels

    # Write into a file
    life_expectancy_change.to_excel("cluster_results.xlsx")

    return life_expectancy_change

    # Add the cluster membership information to the dataframe
    life_expectancy_change["labels"] = labels

    # Write into a file
    life_expectancy_change.to_excel("cluster_results.xlsx")

    return life_expectancy_change


# Use function on DataFrame
result_df = kmeans_clustering_life_expectancy("life_expectancy_change")


def plot_cluster_comparison(result_df):
    """
    Plot a grouped bar chart comparing life expectancy 
    and percentage change for different countries within each label.

    Parameters:
    - result_df (pd.DataFrame): DataFrame containing 

    Returns:
    - None (displays the plot).
    """

    # Assuming the columns are named "1971", "% change", and "labels"
    life_expectancy_change = result_df[["1971", "% change", "labels"]].copy()

    # Normalize the data
    scaler = MinMaxScaler()
    life_expectancy_change[
        ["1971", "% change"]] = scaler.fit_transform(life_expectancy_change[
            ["1971", '% change']])

    # Group the data by 'labels'
    grouped_data = life_expectancy_change.groupby('labels')

    # Define bar width and positions
    bar_width = 0.35
    labels = sorted(life_expectancy_change["labels"].unique())

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(labels), figsize=(15, 5),
                             sharey=True)

    # Custom color palette
    colors = ["#FFA07A", "#800020"]

    for label, ax in zip(labels, axes):
        data_label = grouped_data.get_group(label)
        countries = data_label.index
        life_expectancy = data_label["1971"]
        percentage_change = data_label["% change"]

        r1 = np.arange(len(countries))

        # Create grouped bar chart with custom colors
        bars_life = ax.bar(r1, life_expectancy, color=colors[0],
                           width=bar_width,
                           edgecolor="grey", label="Life Expectancy")
        bars_percentage = ax.bar(r1, percentage_change,
                                 bottom=life_expectancy,
                                 color=colors[1], width=bar_width,
                                 edgecolor="grey", label="% Change")

        # Set labels and title
        ax.set_xlabel("Countries")
        ax.set_title(f"Label {label}")

        # Set x-axis ticks and labels
        ax.set_xticks(np.arange(len(countries)))
        ax.set_xticklabels(countries, rotation=45, ha="right")

    # Set common y-axis label
    fig.text(0, 0.5, "Normalized Values", va="center", rotation="vertical")

    # Create a single legend for the entire plot
    fig.legend([bars_life, bars_percentage],
               ["Life Expectancy", "% Change"], loc="upper right")

    # Set the overall title
    fig.suptitle(
        "Comparison of Cluster Labels: Life Expectancy and % Change (50yrs)",
        fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the plot
    plt.show()

    return


# Use function on DataFrame
plot_cluster_comparison(result_df)


""" Using one country from each of the 3 clusters for fitting. Angola was 
chosen for cluster 0 which had low life expectancy but has experienced 
an increase over a 5o year period. For same period , China was selected 
from Cluster2 (fairly low life expectancy and average % increase ) and 
Switzerland was chosen from cluster 1
 ( low Percent change and high life expectancy)
 """


# Extract data from 1971 onwards to use for clustering
df_china = pd.DataFrame(clean_df_t.loc["China", "1971":"2021"])

df_angola = pd.DataFrame(clean_df_t.loc["Angola", "1971":"2021"])

df_switzerland = pd.DataFrame(clean_df_t.loc["Switzerland", "1971":"2021"])


def fit_exponential_growth(df, column_name, p0=None, output_excel=None):
    """
    Fit an exponential growth curve to a given pandas DataFrame and
    display the results.

    Args:
    - df (pandas DataFrame): The input DataFrame where 'Year' is the index.
    - column_name (str): The name of the column containing the series data.
    - p0 (list, optional): Initial guess for the parameters scale and growth.
    - output_excel (str, optional): File path to save the forecasting results 
    to an Excel file.

    Returns:
    None
    """
    print(df.columns)
    # Extracting time and series data
    year = df.index
    year = year.astype(int)
    series = df[column_name]
    print(type(year))
    # Define the exponential growth function

    def exp_growth(t, scale, growth):
        t = t - 1971  # Adjusting the starting point based on the minimum year
        f = scale * np.exp(growth * t)
        return f

    # If p0 is not provided, use a default initial guess
    if p0 is None:
        p0 = [series.min(), 0.03]

    # Fit exponential growth
    popt, pcov = opt.curve_fit(exp_growth, year, series, p0=p0)

    # Create a DataFrame for forecasting results
    forecast_years = np.arange(2022, 2051)  # Adjust as needed
    forecast_data = exp_growth(forecast_years, *popt)
    forecast_df = pd.DataFrame(
        {'Year': forecast_years, column_name: forecast_data})
    forecast_df.set_index('Year', inplace=True)

    # Save the forecasting results to Excel if specified
    if output_excel is not None:
        forecast_df.to_excel(output_excel)
        print(f"Forecasting results saved to {output_excel}")

    # Plot the data and the fitted exponential growth
    plt.figure()
    plt.plot(year, series, label="Data")
    plt.plot(year, exp_growth(year, *popt), label="Exponential Growth Fit")
    plt.plot(forecast_years, forecast_data, label="Forecast")
    plt.legend()
    plt.title(f"Exponential Growth Fit to {column_name} Data with Forecasting")
    plt.xlabel("Year")
    plt.ylabel(column_name)
    plt.show()

    # Display the fitted parameters
    print(f"Fit parameters: {popt}")
    print(f"Projected {column_name} in 2031: {exp_growth(2031, *popt)}")
    print(f"Projected {column_name} in 2041: {exp_growth(2041, *popt)}")
    print(f"Projected {column_name} in 2051: {exp_growth(2051, *popt)}")

# Example usage:
# Assuming you have DataFrames named df_china, df_angola, df_switzerland


# Call the function for China with a specific initial guess and save to Excel
fit_exponential_growth(df_china, "China", output_excel="china_forecast.xlsx")

# Call the function for Angola with different intial guess and save to Excel
fit_exponential_growth(df_angola, "Angola", p0=[
                       70, 0.09], output_excel="angola_forecast.xlsx")

# Call the function for Switzerland with different  guess and save to Excel
fit_exponential_growth(df_switzerland, "Switzerland", p0=[
                       80, 0.02], output_excel="switzerland_forecast.xlsx")
