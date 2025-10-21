import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import utils.algorithms_utils as algorithms_utils

from typing import Any

import utils.algorithms_utils as algorithm_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import GeneralILPmodel


class WeightedSumAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weighted Sum algorithm'
    
    @staticmethod
    def get_description() -> str:
        return "It obtains supported ILP solutions based on the given weights."

    @staticmethod
    def execute(data_dict: dict, tau: int, info_dict: dict) -> tuple[list[list[str | Any]], Any,list[Any]] | \
                                                                         tuple[list[list[str]], Any | None, None]:

        num_of_objectives = info_dict.get("num_of_objectives")

        objectives_names = info_dict.get("objectives_list")
        model = GeneralILPmodel(active_objectives=objectives_names)
        objectives_list = algorithms_utils.organize_objectives(model, objectives_names)

        subdivisions = info_dict.get("subdivisions")
        weights = info_dict.get("weights")


        if num_of_objectives == 2:
            return weighted_sum_for_two_objectives(model, data_dict, tau, objectives_list, subdivisions, weights)
        elif num_of_objectives == 3:
            return weighted_sum_for_three_objectives(model, data_dict, tau, objectives_list, subdivisions, weights)
        else:
            sys.exit("Number of objectives for weighted sum algorithm must be 2 or 3.")




def weighted_sum_for_two_objectives(model: pyo.AbstractModel, data_dict: dict, tau: int, objectives_list: list,
                                    subdivisions: int=None, weights: tuple=None):
    data = data_dict['data']

    if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
        objectives_list = [model.extractions_objective,
                           model.cc_difference_objective]

    obj1, obj2 = objectives_list[:2]

    csv_data = [[f"Weight1_{obj1.__name__}",f"Weight2_{obj2.__name__}",obj1.__name__,obj2.__name__]]

    output_data = []

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold

    if subdivisions:
        print(f"Processing all ILP results with {subdivisions} subdivisions...")

        for i in range(subdivisions+1):
            w1, w2 = algorithm_utils.generate_two_weights(subdivisions, i)

            algorithm_utils.modify_component(model, 'obj', pyo.Objective(
                rule=lambda m: model.weighted_sum_2obj(m, w1, w2, obj1, obj2)))
            concrete, results = algorithm_utils.concrete_and_solve_model(model, data)

            newrow = [pyo.value(obj1(concrete)), pyo.value(obj2(concrete))]


            algorithm_utils.add_info_to_list(concrete, output_data, results.solver.status, obj1, obj2, newrow)

            newrow = [round(w1,3),round(w2,3)] + newrow
            csv_data.append(newrow)

    elif weights:
        print(f"Processing ILP results with weights: {weights}...")
        w1, w2 = weights

        algorithm_utils.modify_component(model, 'obj', pyo.Objective(
            rule=lambda m: model.weighted_sum_2obj(m, w1, w2, obj1, obj2)))
        concrete, results = algorithm_utils.concrete_and_solve_model(model, data)

        newrow = [pyo.value(obj1(concrete)), pyo.value(obj2(concrete))]
        csv_data.append(newrow)

        algorithm_utils.add_info_to_list(concrete, output_data, results.solver.status, obj1, obj2, newrow)


    else:
        sys.exit(f'The Weighted Sum algorithm parameters for two objectives must be a'
                 f' number of subdivisions s and the second objective.')

    return csv_data, concrete, output_data

def weighted_sum_for_three_objectives(model: pyo.AbstractModel, data_dict: dict, tau: int, objectives_list: list=None,
                                      subdivisions: int=None, weights: tuple=None):

    data = data_dict['data']

    if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
        objectives_list = [model.extractions_objective,
                           model.cc_difference_objective,
                           model.loc_difference_objective]

    obj1, obj2, obj3 = objectives_list[:3]

    # Define threshold
    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

    objectives_names = [obj.__name__ for obj in objectives_list]
    csv_data = [[f"Weight1_{obj1.__name__}", f"Weight2_{obj2.__name__}", f"Weight1_{obj3.__name__}"] + objectives_names]

    concrete = None

    """ Weighted model for all generated weights given a number of subdivisions """
    if subdivisions:
        print(f"Processing all ILP results with {subdivisions} subdivisions")

        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1):
                w1, w2, w3 = algorithm_utils.generate_three_weights(subdivisions, i, j)

                _, newrow, _ = process_weighted_model(model, data, w1, w2, w3, obj1, obj2, obj3)

                csv_data.append(newrow)

                if i == 0:
                    break

        """ Weighted model just for a specific given weights """
    elif weights:
        print(f"Processing the optimal ILP solution with weights: {weights}")

        w1, w2, w3 = weights

        concrete, newrow, results = process_weighted_model(model, data, w1, w2, w3, obj1, obj2, obj3)

        csv_data.append(newrow)

        algorithm_utils.print_result_and_sequences(concrete, results.solver.status, newrow)

    else:
        sys.exit(f'The Weighted Sum algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')

    return csv_data, concrete, None


def process_weighted_model(model: pyo.AbstractModel, data: dp.DataPortal, w1, w2, w3, obj1, obj2, obj3):
    algorithm_utils.modify_component(model, 'obj',
                                     pyo.Objective(rule=lambda m: model.weighted_sum(m, w1, w2, w3, obj1, obj2, obj3)))
    concrete, results = algorithm_utils.concrete_and_solve_model(model, data)

    newrow_values = [pyo.value(obj1(concrete)), pyo.value(obj2(concrete)), pyo.value(obj3(concrete))]
    newrow = [round(w1, 2), round(w2, 2), round(w3, 2)] + newrow_values

    # TODO: añadir generación de CSVs con los resultados (hay algún método ya hecho creo que sería solo llamarlo)

    return concrete, newrow, results
    




