# Import the required libraries
import pandas as pd


# Read the data from a CSV file
def dataframe_from_csv_file(file_path: str) -> pd.DataFrame:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, quotechar='"', delimiter=",", skipinitialspace=True)

    # Strip leading and trailing white spaces from column names
    data.columns = data.columns.str.strip()

    return data


# Save a DataFrame into a CSV file
def dataframe_into_csv_file(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)


# Generate a new DataFrame from the existing one with specified columns and new column names
def get_columns_from_dataframe(df: pd.DataFrame, columns: list, new_column_names: list) -> pd.DataFrame:
    new_df = df[columns].copy()
    new_df.columns = new_column_names
    return new_df
