import os
import sys
import csv
import argparse
import configparser
from pathlib import Path


from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES

from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain
from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel 
import algorithms_utils

# import algorithms_utils
    
# code_filepath: str, model: pyo.AbstractModel, algorithm: str = None, subdivisions: int = None

def main_one_obj(alg_name: str, project_folder: str=None, tau: int=15):
    
    # Uso del script
    # file_path = "GENERAL_results.xlsx"  # Reemplaza con la ruta real del archivo
    # optimal_tuples = algorithms_utils.extract_optimal_tuples(file_path)

    model_engine = ILPEngine()
    model = ILPmodelRsain()
    
    csv_data = [["project", "class", "method", "missingFile", "emptyFile",
         "numberOfSequences", "numberOfVariables", "numberOfConstraints",
         "initialComplexity", "solution", "offsets", "extractions",
         "NOTnestedSolution", "NOTnestedExtractions",
         "NESTEDsolution", "NESTEDextractions",
         "reductionComplexity", "finalComplexity",
         "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC", 
         "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
         "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
         "modelStatus", "terminationCondition", "executionTime"]]

    
    
    instance_folder = "original_code_data"

    for project_folder in sorted(os.listdir(instance_folder)):
        project_folder = Path(project_folder)
        print(f"Project folder: {project_folder}")
        total_path = instance_folder / project_folder
        for class_folder in sorted(os.listdir(total_path)):
            class_folder = Path(class_folder)
            print(f"Class folder: {class_folder}")
            total_path = instance_folder / project_folder / class_folder
            for method_folder in sorted(os.listdir(total_path)):
                method_folder = Path(method_folder)
                print(f"Method folder: {method_folder}")
                total_path = instance_folder / project_folder / class_folder / method_folder
                print(f"Total path: {total_path}")
                if os.path.isdir(total_path):
                    project_folder_name = project_folder.name
                    print(f"Processing project: {project_folder_name}, class: {class_folder}, method: {method_folder}")
                    
                    # folder_tuple = (project_folder_name, class_folder.name, method_folder.name)

                    # if folder_tuple in optimal_tuples:

                    # Process ilp model
                    ilp_model = model.define_model()
                    
                    # Process algorithm
                    algorithm = model_engine.get_algorithm_from_name(alg_name)
                    
                    # Process instance
                    instance = model_engine.load_concrete(total_path, model)
                    
                    folders_data = {"project": str(project_folder_name), "class": str(class_folder), "method": str(method_folder)}
                    results_csv = model_engine.apply_rsain_model(algorithm, ilp_model, instance, tau, csv_data, folders_data)


    # Escribir datos en un archivo CSV
    with open(f"{project_folder_name}_results.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(results_csv)
    
    print("Archivo CSV creado correctamente.")




def main_multiobjective(alg_name: str, instance_folder: Path, tau: int=15, subdivisions: tuple=None, weights: tuple=None, objectives: tuple=None):

    model_engine = ILPEngine()
    model = MultiobjectiveILPmodel()

    if objectives:
        print(f"The objectives are: {objectives}")

        objective_map = {
            'SEQ': model.sequences_objective,
            'CC': model.cc_difference_objective,
            'LOC': model.loc_difference_objective
        }

        try:
            objectives_list = [objective_map[obj.upper()] for obj in objectives]
        except KeyError as e:
            sys.exit(f"Unknown objective '{e.args[0]}'. Objectives must be: SEQ, CC or LOC.")
    else:
        objectives_list = None


    # Process algorithm
    algorithm = model_engine.get_algorithm_from_name(alg_name)
    
    # Process instance
    instance = model_engine.load_concrete(instance_folder, model)
    
    csv_data, concrete_model, output_data, complete_data = model_engine.apply_algorithm(algorithm, instance, tau,
                                                                         subdivisions, weights, objectives_list)

    method_name = os.path.basename(instance_folder)
    class_name = os.path.basename(instance_folder.parent)
    project_name = os.path.basename(instance_folder.parent.parent)

    algorithms_utils.write_output_to_files(csv_data, concrete_model, project_name, class_name, method_name,
                                           alg_name, output_data, complete_data)



PROPERTIES_FILE = "properties.ini"

def delete_ini(path):
    if os.path.exists(path):
        os.remove(path)

def load_config(file=PROPERTIES_FILE):
    """Loads configuration from a file .ini if it exists."""

    config = configparser.ConfigParser()
    config.read(file)

    parameters = {}

    if "Properties" in config:
        section = config["Properties"]
        parameters["model_instance"] = section["model_instance"]
        if "ilp_algorithm" in section:
            parameters["ilp_algorithm"] = section["ilp_algorithm"]
        if "threshold" in section:
            parameters["threshold"] = section.getint("threshold")
        if "subdivisions" in section:
            parameters["subdivisions"] = section.getint("subdivisions")
        if "weights" in section:
            parameters["weights"] = section["weights"]
        if "second_obj" in section:
            parameters["second_obj"] = section["second_obj"]

    return parameters





def save_config(parameters, file=PROPERTIES_FILE):
    """Saves properties in a .ini file"""
    
    
    config = configparser.ConfigParser()
    # Convert lists/tuples into string before saving them
    config["Properties"] = {
        key: ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        for key, value in parameters.items()
    }

    with open(file, "w") as f:
        config.write(f)

    print(f"Properties saved in {file}")
    
    
    
    

def obtain_arguments():
    """Defines arguments from command line and parse them."""

    parser = argparse.ArgumentParser(description='ILP model engine. Given an abstract model m, a model instance a, an algorithm a and optionally a determined number of subdivisions s or three weights w, it applies the correspondent algorithm to find the optimal solutions of the model instance. One can also give as input a properties file path.')
    parser.add_argument('-f', '--file', dest='properties_file', type=str, default=None, help=f'Properties file name in case one want to give every parameter from a .ini file.')
    parser.add_argument('-m', '--modeltype', dest='model_type', type=str, default=None, help='Type of model (uniobjective or multiobjective) used to solve the specific instance.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, default=None, help='Model instance to be optimized (name of the folder with the three data files in CSV format).')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, default=None, help=f'Algorithm to be applied to the model instance {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-t', '--tau', dest='threshold', type=int, default=None, help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=int, default=None, help=f'Number of subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='weights', type=str, default=None, help=f'Weights assigned for weighted sum in the case of a specific combination of weights. Three weights w1,w2,w3 separated by comma (",").')
    parser.add_argument('-o', '--objectives', dest='objectives', type=str, default=None, help=f'Two objectives to minimize in the case of a two objective ILP. Write the two objectives separated by comma (",").')
    parser.add_argument('--save', action='store_true', help='Save properties in a .ini file')

    
    args = parser.parse_args()
    parameters = vars(args)
    

    return parameters

    








if __name__ == '__main__':
    
    # Obtain arguments from command-line
    args = obtain_arguments()
    
    # Load properties from file if it exists
    config = {}
    if args['properties_file']:
        properties_file_path = Path(args['properties_file'])
        print(f"PROPERTIES FILE PATH: {properties_file_path}")
        if not properties_file_path.is_file():
            sys.exit(f'The model instance must be a .ini file.')
        config = load_config(properties_file_path)
    
    
    model_type = args['model_type'] if args['model_type'] else config.get('model_type')
    model_instance = args['model_instance'] if args['model_instance'] else config.get('model_instance')
    ilp_algorithm = args['ilp_algorithm'] if args['ilp_algorithm'] else config.get('ilp_algorithm')
    threshold = args['threshold'] if args['threshold'] else config.get('threshold')
    subdivisions = args['subdivisions'] if args['subdivisions'] else config.get('subdivisions')
    weights = args['weights'] if args['weights'] else config.get('weights')
    objectives = args['objectives'] if args['objectives'] else config.get('objectives')
    
    # Overwrite .ini file values with commandline values if it exists
    for key, value in args.items():
        if value:  # Solo actualizar si el usuario lo pasó por línea de comandos
            config[key] = value
    
    # Save file if there is '--save'
    if args["save"]:
        save_config(config)

    # Show final properties used
    print("Final configuration:")
    for key, value in config.items():
        print(f"   · {key} = {value}")  
    
    
    if model_instance:
        model_instance = Path(model_instance)
        instance_path = "original_code_data" / model_instance
        print(f"INSTANCE PATH: {instance_path}")

        sequences_file = next((f for f in instance_path.iterdir() if f.name.endswith('_sequences.csv')), None)

        if sequences_file:
            with sequences_file.open(newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                filas = list(reader)
                if len(filas) > 1 and len(filas[1]) > 2:
                    x0_cc_value = int(filas[1][2])
                    print(f"Actual CC: {x0_cc_value}.")
        if x0_cc_value <= threshold:
            sys.exit(f'Objective threshold must be lower than actual CC.')

        if not instance_path.is_dir():
            sys.exit(f'The model instance must be a folder with three CSV files.')
            
    # Turn "w1,w2,w3" into (float,float,float) if --weights is a parameter in command line
    if weights:
        weights = tuple(map(float, weights.split(",")))
        
    # Turn "obj1,obj2" into (str,str) if --objectives is a parameter in command line
    if objectives:
        objectives = tuple(map(str, objectives.split(",")))
    
    
    if model_type == 'uniobjective':
        if model_instance:
            main_one_obj(ilp_algorithm, instance_path, threshold)
        else:
            main_one_obj(ilp_algorithm, threshold)
    elif model_type == 'multiobjective':
        main_multiobjective(ilp_algorithm, instance_path, int(threshold), subdivisions, weights, objectives)
    else:
        sys.exit("No adequate number of parameters have been provided. Run python main.py -h for help.")
    
    
    