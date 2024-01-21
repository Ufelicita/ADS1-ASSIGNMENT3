import pandas as pd
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np


def read_transpose_clean(file):
    """
    Reads, transposes, and cleans the data.
    """

    # Read the data
    df = pd.read_csv(file)

    # Drop the first column (index)
    df = df.iloc[:, 1:]

    # Set the display format for float values
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Transpose dataframe
    df_t = pd.DataFrame.transpose(df)
    print(df_t)

    # Make the first row as header
    df_t.columns = df_t.iloc[0].values.tolist()

    # Delete the first 3 rows as it would not be required
    df_t = df_t.iloc[3:]

    # Remove NAN
    clean_df = df_t.dropna()

    # print to check if NAN  still exists in data frame
    print(clean_df.isnull().sum())

    # Rename Index column header as "Year"
    clean_df.index.names = ["Year"]

    print(clean_df.index)

    # Print to check data type of each column
    print(clean_df.dtypes)

    # Change index columns data type to numeric
    clean_df.index = pd.to_numeric(clean_df.index)
    print(clean_df.index)

    cols_num = clean_df.columns[0:]
    clean_df[cols_num] = clean_df[cols_num].apply(pd.to_numeric)
    print(clean_df.dtypes)

    return clean_df


# Analyse cleaned data
clean_df = read_transpose_clean("API_NE.EXP.GNFS.CD_DS2_en_csv_v2_6300789.csv")

# Summary statistics for numerical data
print("\nNumerical Summary:")
print(clean_df.describe())


def generate_correlation_heatmap(dataframe, selected_years):
    """
    Generate a correlation heatmap and scatter matrix for specific years
    in the dataframe to see which one correlates better and not too correlated.

    Parameters:
    - dataframe: pd.DataFrame
        The main DataFrame containing the data.
    - selected_years: list
        List of years to include in the analysis.

    Returns:
    - None
    """

    # Create a subset of the main DataFrame with selected years
    subset_df = dataframe[dataframe.index.isin(selected_years)]

    # Transpose the subset DataFrame for correct visualization
    subset_df = subset_df.transpose()

    # Convert the DataFrame to numeric (in case there are non-numeric values)
    subset_df = subset_df.apply(pd.to_numeric)

    # Calculate the correlation matrix
    corr_matrix = subset_df.corr(numeric_only=True)

    # Create a heatmap
    plt.figure(figsize=[10, 8])
    plt.imshow(corr_matrix, cmap='viridis')
    plt.colorbar()
    annotations = subset_df.columns[0:12]  # extract relevant headers
    plt.xticks(ticks=range(len(corr_matrix)), labels=annotations)
    plt.yticks(ticks=range(len(corr_matrix)), labels=annotations)
    plt.title('Correlation Heatmap')

    # Save and Show plots
    plt.savefig("heatmap.png")
    plt.show()

    return


generate_correlation_heatmap(clean_df, [1980, 1984, 1988, 1992,
                                        1996, 2000, 2004, 2008,
                                        2012, 2016, 2020])

selected_years = [1980, 2020]
export_df = clean_df[clean_df.index.isin(selected_years)]
export_df_t = export_df.transpose()

export_df_subtract =  export_df_t[2020] -  export_df_t[1980]
export_df_t["Growth"] = 100/40 * export_df_subtract/export_df_t[1980]

print(export_df_t.describe())

plt.figure(figsize=(8, 8))
plt.scatter(export_df_t[1980], export_df_t["Growth"])
plt.xlabel("Export (1980)")
plt.ylabel(" Export Growth per year [%]")
plt.show()


