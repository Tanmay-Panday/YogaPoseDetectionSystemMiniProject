import pandas as pd
import os

def sort_csv_by_class(file_path):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Sort the DataFrame based on the "class" column
        df_sorted = df.sort_values(by='class')

        # Write the sorted DataFrame back to the CSV file
        df_sorted.to_csv(file_path, index=False)

        print(f"Rows in '{file_path}' sorted alphabetically based on 'class' column.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")

currentDirectory = os.path.dirname(os.path.realpath(__file__))
PythonCodesAndDatabase = os.path.dirname(currentDirectory)
coordinates_path = os.path.join(PythonCodesAndDatabase, 'Database', 'coordinates.csv')
sort_csv_by_class(coordinates_path)
