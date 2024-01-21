import pandas as pd
import sklearn.cluster as cl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    df_cleaned = df_t.dropna()
    
    # print to check if NAN  still exists in data frame 
    print(df_cleaned.isnull().sum())
    
    # Rename Index column header as "Year" 
    df_cleaned.index.names = ["Year"]

    print(df_cleaned.index)
    
    # Print to check data type of each column 
    print(df_cleaned.dtypes)
    
    # Change index columns data type to numeric 
    df_cleaned.index = pd.to_numeric(df_cleaned.index)
    print(df_cleaned.index)
    
    cols_num = df_cleaned.columns[0:]
    df_cleaned[cols_num] = df_cleaned[cols_num].apply(pd.to_numeric)
    print(df_cleaned.dtypes)
       
    return df_cleaned


# Analyse cleaned data 
df_cleaned = read_transpose_clean(
    "API_NE.EXP.GNFS.CD_DS2_en_csv_v2_6300789.csv")

# Summary statistics for numerical data
print("\nNumerical Summary:")
print(df_cleaned.describe())


# Specify the years for correlations
select_years = [1988,1992, 2020]

# Create a subset of the main DataFrame with selected years
subset_df = df_cleaned[df_cleaned.index.isin(select_years)]
# Transpose the subset DataFrame for correct visualization
subset_df = subset_df.transpose()

subset_df = subset_df.apply(pd.to_numeric)

print()
print(subset_df .dtypes)

corr = subset_df.corr(numeric_only=True)
print(corr.round(4))


# fig, ax = plt.subplots(figsize=(8, 8))
plt.figure(figsize=[10, 8])
# this prouces an image
plt.imshow(corr)
plt.colorbar()
annotations = subset_df.columns[0:2] # extract relevant headers
plt.xticks(ticks=[0, 1,], labels=annotations)
plt.yticks(ticks=[0, 1,], labels=annotations)
# It can be saved as other graphs
plt.savefig("heatmap.png")
plt.show()

pd.plotting.scatter_matrix(subset_df, figsize=(10, 10), s=10)
plt.savefig("scatter_matrix.png")
plt.show()


#
