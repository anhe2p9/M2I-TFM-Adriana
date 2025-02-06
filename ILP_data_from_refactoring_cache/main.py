# Import the required libraries
from ILP_data_from_refactoring_cache.utils import dataset as dataset, refactoring_cache as rc
import argparse


# Main function
def main(path_to_refactoring_cache: str, output_folder: str):
    df = rc.set_extractions_id(dataset.dataframe_from_csv_file(path_to_refactoring_cache))

    # Save the extractions in conflict into a CSV file
    dataset.dataframe_into_csv_file(rc.get_conflicts(df), output_folder + "conflicts.csv")

    # Save the nested extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_nested_extraction_for_each_extraction(df, ["reductionCC"]),
                                    output_folder + "nestedExtractions.csv")

    # Save the lines of code and cognitive complexity of the extractions into a CSV file   
    dataset.dataframe_into_csv_file(rc.get_nested_extraction_for_each_extraction_computing_ccr(df),
                                    output_folder + "sequences.csv")


# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process refactoring cache and output results.')
    parser.add_argument('input_file', type=str, help='Path to the refactoring cache CSV file')
    parser.add_argument('output_folder', type=str, help='Folder to save the output CSV files')
    args = parser.parse_args()

    main(args.input_file, args.output_folder)
