# Import the required libraries
from ILP_data_from_refactoring_cache.utils import dataset as dataset, refactoring_cache as rc
import argparse
import os
from pathlib import Path
from fileinput import input
import re

# Main function
def main(path_to_refactoring_cache: str, output_folder: str, files_n: str):
    # Filter feasible extractions and assign id to extractions
    df = rc.set_extractions_id(dataset.dataframe_from_csv_file(path_to_refactoring_cache))

    # Save the mapping between feasible extractions and offsets into a CSV file
    dataset.dataframe_into_csv_file(
        rc.get_extractions_including_given_columns(df, ["A", "B"]), output_folder + f"/{files_n}_feasible_extractions_offsets.csv")

    # Save the extractions in conflict into a CSV file
    dataset.dataframe_into_csv_file(rc.get_conflicts(df), output_folder + f"/{files_n}_conflict.csv")

    # Save the lines of code and cognitive complexity of the extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_extractions_including_given_columns(df, ["extractedLOC", "extractedMethodCC"]),
                                    output_folder + f"/{files_n}_sequences.csv")

    # Save the nested extractions into a CSV file
    dataset.dataframe_into_csv_file(rc.get_nested_extraction_for_each_extraction_computing_ccr(df),
                                    output_folder + f"/{files_n}_nested.csv")
    
    
    
def extraer_clase_metodo(nombre_archivo):
        patron = r'\.(?P<clase>[^.]+)\.java\.(?P<metodo>[^.]+)\.csv'
        coincidencia = re.search(patron, nombre_archivo)
        
        if coincidencia:
            return coincidencia.group("clase"), coincidencia.group("metodo")
        return None, None



# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process refactoring cache and output results.')
    parser.add_argument('input_folder', type=str, help='Path to the refactoring cache CSV file')
    parser.add_argument('output_folder', type=str, help='Folder to save the output CSV files')
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    
    for file in input_folder.iterdir():
        file_class, file_method = extraer_clase_metodo(str(file))
        # print(f"Clase: {file_class}, Método: {file_method}")
    
        # Base directory of the project
        base_dir = Path(__file__).resolve().parent.parent
        
        # Build Path
        output_dir = base_dir / "original_code_data" / args.output_folder / file_class / file_method
        
        # Create folder if it does not exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        main(file, str(output_dir), file_method)
        print(f"New data is available in: {output_dir}")

