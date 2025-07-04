import os
import os.path

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

import sys
import csv
import argparse
import configparser
from pathlib import Path

from grapheme import length

import utils.results_utils as results_utils

from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES

from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain
from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel

model_engine = ILPEngine()
model = ILPmodelRsain()
multiobjective_model = MultiobjectiveILPmodel()

def main_one_obj(alg_name: str, instance_path: Path=None, tau: int=15, objective: str=None):

    if objective:
        print(f"The objective is: {objective}")

        objective_map = {
            'EXTRACTIONS': model.extractions_objective,
            'CC': model.cc_difference_objective,
            'LOC': model.loc_difference_objective
        }

        try:
            objective= next(objective_map[obj.upper()] for obj in objectives)
        except KeyError as e:
            sys.exit(f"Unknown objective '{e.args[0]}'. Objectives must be: EXTRACTIONS, CC or LOC.")
    
    csv_data = ["project", "class", "method", "missingFile", "emptyFile",
         "numberOfSequences", "numberOfVariables", "numberOfConstraints",
         "initialComplexity", "solution", "offsets", "extractions",
         "NOTnestedSolution", "NOTnestedExtractions",
         "NESTEDsolution", "NESTEDextractions",
         "reductionComplexity", "finalComplexity",
         "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC", 
         "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
         "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
         "modelStatus", "terminationCondition", "executionTime"]

    # Crear el archivo desde cero (sobrescribir si existe)
    csv_path = f"{instance_path}_{objective.__name__}_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data)
        writer.writeheader()

    for project_folder in sorted(os.listdir(instance_path)):
        project_folder = Path(project_folder)
        print(f"Project folder: {project_folder}")
        total_path = instance_path / project_folder
        for class_folder in sorted(os.listdir(total_path)):
            class_folder = Path(class_folder)
            print(f"Class folder: {class_folder}")
            total_path = instance_path / project_folder / class_folder
            for method_folder in sorted(os.listdir(total_path)):
                method_folder = Path(method_folder)
                print(f"Method folder: {method_folder}")
                total_path = instance_path / project_folder / class_folder / method_folder
                print(f"Total path: {total_path}")
                if os.path.isdir(total_path):
                    project_folder_name = project_folder.name
                    print(f"Processing project: {project_folder_name}, class: {class_folder}, method: {method_folder}")

                    # Check threshold
                    check_threshold(total_path)

                    # Process algorithm
                    algorithm = model_engine.get_algorithm_from_name(alg_name)
                    
                    # Process instance
                    instance = model_engine.load_concrete(total_path, model)
                    
                    folders_data = {
                        "project": str(project_folder_name),
                        "class": str(class_folder),
                        "method": str(method_folder)
                                    }

                    if objective.__name__ == 'extractions_objective':
                        results_csv = model_engine.apply_rsain_model(algorithm,instance, tau, folders_data, objective)
                    else:
                        results_csv = model_engine.apply_algorithm(algorithm, instance, tau, folders_data,
                                                                   objective)

                    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(results_csv)
                    print("Added line succesfully.")
                    print("============================================================================================================")
    
    print("CSV file with results for one objective correctly created.")




def main_multiobjective(num_of_objectives: int, alg_name: str, instance_folder: Path, tau: int=15,
                        subdivisions: tuple=None, weights: tuple=None, objectives: tuple=None):

    if objectives:
        print(f"The objectives are: {objectives}")

        objective_map = {
            'EXTRACTIONS': multiobjective_model.extractions_objective,
            'CC': multiobjective_model.cc_difference_objective,
            'LOC': multiobjective_model.loc_difference_objective
        }

        try:
            objectives_list = [objective_map[obj.upper()] for obj in objectives]
        except KeyError as e:
            sys.exit(f"Unknown objective '{e.args[0]}'. Objectives must be: EXTRACTIONS, CC or LOC.")
    else:
        objectives_list = None


    # Process algorithm
    algorithm = model_engine.get_algorithm_from_name(alg_name)
    
    # Process instance
    instance = model_engine.load_concrete(instance_folder, multiobjective_model)

    complete_data, nadir = None, None

    if alg_name == 'WeightedSumAlgorithm':
        csv_data, concrete_model, output_data = model_engine.apply_algorithm(algorithm, instance,tau,
                                                                             num_of_objectives, subdivisions,
                                                                             weights, objectives_list)
    elif alg_name == 'EpsilonConstraintAlgorithm' or alg_name == 'HybridMethodForThreeObj':
        csv_data, concrete_model, output_data, complete_data, nadir = model_engine.apply_algorithm(algorithm,
                                                                                                   instance,
                                                                                                   tau,
                                                                                                   objectives_list)
    else:
        sys.exit(f"Unknown algorithm '{alg_name}'. Algorithms for more than one objective must be:"
                 f" WeightedSumAlgorithm, EpsilonConstraintAlgorithm, or HybridMethodForThreeObj.")

    method_name, class_name, project_name = get_all_path_names(instance_folder)

    write_output_to_files(csv_data, concrete_model, project_name, class_name, method_name,
                                           alg_name, output_data, complete_data, nadir)



def get_all_path_names(instance_folder: Path):
    method_name = os.path.basename(instance_folder)
    class_name = os.path.basename(instance_folder.parent)
    project_name = os.path.basename(instance_folder.parent.parent)
    return method_name, class_name, project_name


def write_output_to_files(csv_info: list, concrete: pyo.ConcreteModel, project_name: str, class_name: str,
                          method_name: str, algorithm: str, output_data: list = None, complete_data: list = None,
                          nadir: list = None):
    result_name = f"{algorithm}_{class_name}_{method_name}"
    base_path = f"output/results/{project_name}/{result_name}"
    method_path = f"{base_path}/{method_name}"

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Save model in a LP file
    if concrete:
        concrete.write(f'{method_path}.lp',
                       io_options={'symbolic_solver_labels': True})
        print("Model correctly saved in a LP file.")

    # Save data in a CSV file
    filename = f"{method_path}_results.csv"

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_info)
        print("CSV file correctly created.")

    # Save output in a TXT file
    if output_data:
        with open(f"{method_path}_output.txt", "w") as f:
            for linea in output_data:
                f.write(linea + "\n")
            print("Output correctly saved in a TXT file.")

    # Save output in a TXT file
    if complete_data:
        with open(f"{method_path}_complete_data.csv",
                  mode="w", newline="", encoding="utf-8") as complete_csv:
            writer = csv.writer(complete_csv)
            writer.writerows(complete_data)
            print("Complete CSV file correctly created.")

    # # Save nadir point in a csv file
    # if nadir:
    #     with open(f"{method_path}_nadir.csv",
    #               mode="w", newline="", encoding="utf-8") as nadir_csv:
    #         writer = csv.writer(nadir_csv)
    #         writer.writerows(nadir)
    #         print("Nadir CSV file correctly created.")



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


def check_threshold(model_instance):
    model_instance = Path(model_instance)
    print(f"INSTANCE PATH: {model_instance}")

    sequences_file = next((f for f in model_instance.iterdir() if f.name.endswith('_sequences.csv')), None)
    if sequences_file:
        with sequences_file.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            filas = list(reader)
            if len(filas) > 1 and len(filas[1]) > 2:
                x0_cc_value = int(filas[1][2])
                print(f"Actual CC: {x0_cc_value}.")
        if x0_cc_value <= threshold:
            sys.exit(f'Objective threshold must be lower than actual CC.')

    if not model_instance.is_dir():
        sys.exit(f'The model instance must be a folder with three CSV files (multiobjective)'
                 f' or the base path with all projects (one objective).')
    
    
    
    

def obtain_arguments():
    """Defines arguments from command line and parse them."""

    parser = argparse.ArgumentParser(
        description='ILP model engine. Given an abstract model m, a model instance a, an algorithm a and optionally '
                    'a threshold t, '
                    'a determined number of subdivisions s or three weights w,'
                    'an objectives order o, and other additional argumentes,'
                    ' it applies the correspondent algorithm to find the optimal solutions of the model instance. '
                    'One can also give as input a properties file path.')
    parser.add_argument('-f', '--file', dest='properties_file', type=str, default=None,
                        help=f'Properties file name in case one want to give every parameter from a .ini file.')
    parser.add_argument('-n', '--num_of_objectives', dest='num_of_objectives', type=str, default=None,
                        help=f'Number of objectives to minimize.')
    # parser.add_argument('-m', '--modeltype', dest='model_type', type=str, default=None,
    #                     help='Type of model (uniobjective or multiobjective) used to solve the specific instance.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, default=None,
                        help='Model instance to be optimized. '
                             'It can be the folder path with the three data files in CSV format for multiobjective'
                             'or the general folder path with all instances for one objective.')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, default=None,
                        help=f'Algorithm to be applied to the model instance in the case of multiobjective ILP:'
                             f' {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-t', '--tau', dest='threshold', type=int, default=None,
                        help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=int,
                        default=None, help=f'Number of subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='weights', type=str, default=None,
                        help=f'Weights assigned for weighted sum in the case of a specific combination of weights.'
                             f' Three weights w1,w2,w3 separated by comma (",").')
    parser.add_argument('-o', '--objectives', dest='objectives', type=str, default=None,
                        help=f'List of objectives to minimize. '
                             f'In case of two or three objectives, write them separated by comma (","):'
                             f' "obj1", "obj1,obj2" or "obj1,obj2,ob3".')
    parser.add_argument('--plot', action='store_true',
                        help=f'Plots the result of the given result. It gives just one plot.')
    parser.add_argument('--3dPF', action='store_true',
                        help=f'Plots the 3D PF of the given result. It gives just one PF plot.')
    parser.add_argument( '--all_plots', action='store_true',
                        help=f'Plots all results in a given directory. More than one plot will be created.')
    parser.add_argument('--statistics', action='store_true',
                        help=f'Creates a CSV file with the statistics of all the results found in a given directory.'
                             f'The statistics are: hypervolume, median, iqr, average and std for each objective.')
    parser.add_argument('--all_3dPF', action='store_true',
                        help=f'Plots all 3D PFs in a given directory. More than one PF plot will be created.')
    parser.add_argument('--input', dest='input_dir', type=str, default=None,
                        help=f'The input path for plots and/or statistics can be specified,'
                             f' and if there is no input path, the output path will be the general "output/results" '
                             f'folder for all results.')
    parser.add_argument('--output', dest='output_dir', type=str, default=None,
                        help=f'The output path for plots and/or statistics can be specified,'
                             f' and if there is no output path, the output path will be the general'
                             f' "output/plots_and_statistics" folder for all results.')
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
        properties_file_path = args['properties_file']
        print(f"PROPERTIES FILE PATH: {properties_file_path}")
        if not properties_file_path.is_file():
            sys.exit(f'The model instance must be a .ini file.')
        config = load_config(properties_file_path)
    
    
    # model_type = args['model_type'] if args['model_type'] else config.get('model_type')
    num_of_objectives = int(args['num_of_objectives']) if args['num_of_objectives'] else config.get('num_of_objectives')
    model_instance = args['model_instance'] if args['model_instance'] else config.get('model_instance')
    ilp_algorithm = args['ilp_algorithm'] if args['ilp_algorithm'] else config.get('ilp_algorithm')
    threshold = args['threshold'] if args['threshold'] else config.get('threshold')
    subdivisions = args['subdivisions'] if args['subdivisions'] else config.get('subdivisions')
    weights = args['weights'] if args['weights'] else config.get('weights')
    objectives = args['objectives'] if args['objectives'] else config.get('objectives')
    input_dir = args['input_dir'] if args['input_dir'] else config.get('input_dir')
    output_dir = args['output_dir'] if args['output_dir'] else config.get('output_dir')

    # Check that there is number of objectives specified
    if model_instance and not num_of_objectives:
        sys.exit(f'No number of objectives specified, please specify the number of objectives to minimize.'
                 f' Type python main.py -h for help.')
    
    # Overwrite .ini file values with commandline values if it exists
    for key, value in args.items():
        if value:  # Solo actualizar si el usuario lo pasó por línea de comandos
            config[key] = value
    
    # Save file if there is '--save'
    if args["save"]:
        save_config(config)

    # Check model instance
    if model_instance:
        instance_path = Path(model_instance)

    # Show final properties used
    print("Final configuration:")
    for key, value in config.items():
        print(f"   · {key} = {value}")
            
    # Turn "w1,w2,w3" into (float,float,float) if --weights is a parameter in command line
    if weights:
        weights = tuple(map(float, weights.split(",")))

    # Turn "obj1,obj2" into (str,str) if --objectives is a parameter in command line
    if objectives:
        objectives = tuple(map(str, objectives.split(",")))
        if len(objectives) != num_of_objectives:
            sys.exit("The length of the objectives list must be the same as the number of objectives specified.")

    # Single plot True if there is --single_plot
    if args["plot"]:
        single_plot = True
    else:
        single_plot = False

    # Single 3D PF plot True if there is --3dPF
    if args["3dPF"]:
        single_3D_PF = True
    else:
        single_3D_PF = False

    # All plots True if there is --all_plots
    if args["all_plots"]:
        all_plots = True
    else:
        all_plots = False

    # All plots True if there is --all_plots
    if args["all_3dPF"]:
        all_3DPF = True
    else:
        all_3DPF = False

    # Statistics True if there is --statistics
    if args["statistics"]:
        statistics = True
    else:
        statistics = False

    # Input files
    if not input_dir:
        input_dir = "output/results"

    # Output files
    if not output_dir and input_dir:
        input_path = Path(input_dir)
        output_dir = input_path.parent
    elif not output_dir and not input_dir:
        output_dir = "output/plots_and_statistics"


    if num_of_objectives:
        if num_of_objectives == 1:
            if model_instance:
                check_threshold(model_instance)
                if not ilp_algorithm:
                    ilp_algorithm = 'ObtainResultsAlgorithm'
                main_one_obj(ilp_algorithm, model_instance, int(threshold),objectives)
            else:
                sys.exit('General instance folder required.')
        elif num_of_objectives > 1 and model_instance:
            check_threshold(model_instance)
            main_multiobjective(num_of_objectives, ilp_algorithm, instance_path, int(threshold), subdivisions, weights, objectives)

            method_name, class_name, project_name = get_all_path_names(instance_path)
            general_path = f"{project_name}/{ilp_algorithm}_{class_name}_{method_name}/{method_name}"

            input_general_path = f"output/results/{general_path}"
            results_csv_path = f"{input_general_path}_results.csv"
            single_3D_PF_path = f"{input_general_path}_3DPF.html"

            if single_plot:
                if num_of_objectives == 2:
                    single_plot_path = f"{input_general_path}_2DPF_plot.pdf"
                    results_utils.generate_2DPF_plot(results_csv_path, single_plot_path)
                elif num_of_objectives == 3:
                    single_plot_path = f"{input_general_path}_parallel_coordinates_plot.pdf"
                    results_utils.generate_plot(results_csv_path, single_plot_path)
            if single_3D_PF:
                results_utils.generate_3DPF_plot(results_csv_path, single_3D_PF_path)

    if all_plots:
        results_utils.traverse_and_plot(input_dir, output_dir)

    if statistics:
        results_utils.generate_statistics(input_dir, output_dir)

    if all_3DPF:
        results_utils.traverse_and_PF_plot(input_dir, output_dir)

    
    