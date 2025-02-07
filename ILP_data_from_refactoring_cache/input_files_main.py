# Import the required libraries
from ILP_data_from_refactoring_cache.utils import dataset as dataset, refactoring_cache as rc
import argparse
import os
from pathlib import Path

# Main function
def main(path_to_refactoring_cache: str, output_folder: str, files_n: str):
    # Filter feasible extractions and assign id to extractions
    df = rc.set_extractions_id(dataset.dataframe_from_csv_file(path_to_refactoring_cache))

    # Save the mapping between feasible extractions and offsets into a CSV file
    dataset.dataframe_into_csv_file(
        rc.get_extractions_including_given_columns(df, ["A", "B"]), output_folder + f"/{files_n}feasible_extractions_offsets.csv")

    # Save the extractions in conflict into a CSV file
    dataset.dataframe_into_csv_file(rc.get_conflicts(df), output_folder + f"/{files_n}conflicts.csv")

    # Save the lines of code and cognitive complexity of the extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_extractions_including_given_columns(df, ["extractedLOC", "extractedMethodCC"]),
                                    output_folder + f"/{files_n}_sequences.csv")

    # Save the nested extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_nested_extraction_for_each_extraction_computing_ccr(df),
                                    output_folder + f"/{files_n}_nested.csv")


# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process refactoring cache and output results.')
    parser.add_argument('input_file', type=str, help='Path to the refactoring cache CSV file')
    parser.add_argument('output_folder', type=str, help='Folder to save the output CSV files')
    args = parser.parse_args()

    nested_directory_path = Path("original_code_data/bytecode-viewer")
    
    # Create nested directories
    nested_directory_path.mkdir(parents=True, exist_ok=True)
    
    directory_path_str = str(nested_directory_path)
    
    last_dir = os.path.basename(nested_directory_path)
    
    
    main(args.input_file, directory_path_str, last_dir)

