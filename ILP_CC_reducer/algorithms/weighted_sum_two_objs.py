import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

import sys

from typing import Any

import utils.algorithms_utils as algorithm_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel


class WeightedSumAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(data_dict: dict, tau: int, weights_config: Any, objectives_list: list) -> None:
        
        multiobjective_model = MultiobjectiveILPmodel()
        data = data_dict['data']

        if not objectives_list:  # if there is no order, the order will be [SEQ,CC]
            objectives_list = [multiobjective_model.sequences_objective,
                               multiobjective_model.cc_difference_objective]

        obj1, obj2 = objectives_list[:2]
        
        csv_data = [[f"Weight1_{obj1.__name__}",f"Weight2_{obj2.__name__}",obj1.__name__,obj2.__name__]]
        
        output_data = []
        
        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
        
        if isinstance(weights_config, int):
            print(f"Proccessing all ILP results with {weights_config} subdivisions...")
        
            for i in range(weights_config+1):
                w1, w2 = algorithm_utils.generate_two_weights(weights_config, i)
               
                algorithm_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(rule=lambda m: multiobjective_model.weighted_sum_2obj(m, w1, w2, obj1, obj2)))
                concrete, results = algorithm_utils.concrete_and_solve_model(multiobjective_model, data)
                
                newrow = [pyo.value(obj1(concrete)), pyo.value(obj2(concrete))]
                
                
                algorithm_utils.add_info_to_list(concrete, output_data, results.solver.status, obj1, obj2, newrow)
                
                newrow = [round(w1,3),round(w2,3)] + newrow
                csv_data.append(newrow)
                
        elif all(isinstance(w, float) for w in weights_config):
            print(f"Proccessing ILP results with weights: {weights_config}...")
            w1, w2 = weights_config
            
            algorithm_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(rule=lambda m: multiobjective_model.weighted_sum_2obj(m, w1, w2, obj1, obj2)))
            concrete, results = algorithm_utils.concrete_and_solve_model(multiobjective_model, data)
            
            newrow = [pyo.value(obj1(concrete)), pyo.value(obj2(concrete))]
            csv_data.append(newrow)
            
            algorithm_utils.add_info_to_list(concrete, output_data, results.solver.status, obj1, obj2, newrow)
                
            
        else:
            sys.exit(f'The Weighted Sum algorithm parameters for two objectives must be a number of subdivisions s and the second objective.')
            
        return csv_data, concrete, output_data, None, None



    




