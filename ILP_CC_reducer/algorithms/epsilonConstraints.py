import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

# from typing import Any
# import sys

# import os
# import csv

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import algorithms_utils



class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm with 3 obj'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, objectives_list: list= None):
                
        multiobjective_model = MultiobjectiveILPmodel()

        if not objectives_list:  # if there is no order, the order will be [SEQ,CC,LOC]
            objectives_list = [multiobjective_model.sequences_objective,
                               multiobjective_model.cc_difference_objective,
                               multiobjective_model.loc_difference_objective]

        obj1, obj2, obj3 = objectives_list[:3]
        
        csv_data = [[obj.__name__ for obj in objectives_list]]

        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

        multiobjective_model.obj = pyo.Objective(rule=lambda m: obj2(m))  # Objective {min f2}

        """ z <- Solve {min f2(x) subject to x in X} """
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)
        
        
        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):

            """ z <- Solve {min f3(x) subject to f2(x) <= f2(z)} """
            f2z = pyo.value(concrete.obj)  # f2(z) := f2z
            
            print("=====================================================================================================================================")
            print(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}")

            multiobjective_model.f2z = pyo.Param(
                initialize=f2z)  # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobjective_model.f2Constraint = pyo.Constraint(
                rule=lambda m: obj2(m) <= m.f2z)  # new constraint: f2(x) <= f2(z)


            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: obj3(m)))  # new objective: min f3(x)

            """ Solve {min f3(x) subject to f2(x) <= f2(z)} """
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)
            
            
            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                
                """ z <- Solve {min f1(x) subject to f2(x) <= f2(z), f3(x) <= f3(z)} """
                f3z = pyo.value(concrete.obj)  # f3(z) := f3z

                multiobjective_model.f3z = pyo.Param(
                    initialize=f3z)  # new parameter f3(z) to implement new constraint f3(x) <= f3(z)
                multiobjective_model.f3Constraint = pyo.Constraint(
                    rule=lambda m: obj3(m) <= m.f3z)  # new constraint f3(x) <= f3(z)
                

                algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
                    rule=lambda m: obj1(m)))  # new objective: min f1(x)

                """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)  # Solve model
                multiobjective_model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint
                multiobjective_model.del_component('f3Constraint')  # delete f3(x) <= f3(z) constraint

                f1z = pyo.value(concrete.obj)
                
                print(f"min f1(x), sequences, subject to f2(x) <= f2(z), f3(x) <= f3(z): {f1z}")

    
                """ FP <- {z} (add z to Pareto front) """
                newrow = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
                csv_data.append(newrow)
                
                algorithms_utils.print_result_and_sequences(concrete, result.solver.status, newrow)            

                multiobjective_model.epsilon1 = pyo.Param(initialize=f1z-1, mutable=True)  # epsilon2 <- f1(z) - 1
                multiobjective_model.epsilon2 = pyo.Param(initialize=f2z-1, mutable=True)  # epsilon2 <- f2(z) - 1

                u1 = f1z - 1  # lower bound for f1(x)
                u2 = f2z - 1  # lower bound for f2(x)

                multiobjective_model.l1 = pyo.Var(within=pyo.NonNegativeReals)  # l1 = epsilon1 - f1(x)
                multiobjective_model.l2 = pyo.Var(within=pyo.NonNegativeReals)  # l2 = epsilon2 - f2(x)

                solution_found = (result.solver.status == 'ok') and (
                        result.solver.termination_condition == 'optimal')  # while loop control
                
                while solution_found:

                    multiobjective_model.lambd1 = pyo.Param(initialize=1/(f1z - u1))  # estimate a lambda1 value > 0
                    multiobjective_model.lambd2 = pyo.Param(initialize=1/(f2z - u2))  # estimate a lambda2 value > 0
                    
                    """ Solve {min f3(x) - (lambda1 * l1 + lambda2 * l2), subject to f1(x) + l1 = epsilon1, f2(x) + l2 = epsilon2} """
                    algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
                        rule=lambda m: multiobjective_model.epsilon_objective(m, obj3)))  # min f3(x) - (lambda1 * l1 + lambda2 * l2)
                    

                    algorithms_utils.modify_component(
                        multiobjective_model, 'epsilonConstraint1', pyo.Constraint(
                            rule=lambda m: obj1(m) + m.l1 == m.epsilon1))  # subject to f1(x) + l1 = epsilon1
                    algorithms_utils.modify_component(
                        multiobjective_model, 'epsilonConstraint2', pyo.Constraint(
                            rule=lambda m: obj2(m) + m.l2 == m.epsilon2))  # subject to f2(x) + l2 = epsilon2

                    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)  # Solve

                    result.write()

                    """ While exists x in X that makes f1(x) < epsilon do """
                    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                        
                        print(f"slack variable l1 value: {concrete.l1.value}")
                        print(f"slack variable l2 value: {concrete.l1.value}")
                        
                        """ PF = PF U {z} """
                        newrow = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
                        csv_data.append(newrow)
                        
                        
                        """ epsilon1 = f1(z) - 1 """
                        f1z = pyo.value(multiobjective_model.sequences_objective(concrete))
                        algorithms_utils.modify_component(multiobjective_model, 'epsilon1', pyo.Param(
                            initialize=f1z-1, mutable=True))  # New epsilon1 value
                        
                        
                        """ epsilon2 = f2(z) - 1 """
                        f2z = pyo.value(multiobjective_model.cc_difference_objective(concrete))
                        algorithms_utils.modify_component(multiobjective_model, 'epsilon2', pyo.Param(
                            initialize=f2z-1, mutable=True))  # New epsilon2 value

                        print(f"f1z: {f1z}")
                        print(f"f2z: {f2z}")

                        u1 = f1z - 1  # lower bound for f1(x) (it has to decrease with f1z)
                        u2 = f2z - 1 # lower bound for f2(x) (it has to decrease with f2z)

                        print(f"epsilon1: {concrete.epsilon1.value}")
                        print(f"epsilon2: {concrete.epsilon2.value}")
                        print(f"u1: {u1}")
                        print(f"u2: {u2}")
                        print(f"lambda1: {concrete.lambd1.value}")
                        print(f"lambda2: {concrete.lambd2.value}")
                        print(f"comprobacion1: {f1z} <= {concrete.epsilon1.value}")
                        print(f"comprobacion2: {f2z} <= {concrete.epsilon2.value}")
                        
                        algorithms_utils.print_result_and_sequences(concrete, result.solver.status, newrow)

                    solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')
                
        return csv_data, concrete, None


# def update(meta_list: list[list], z):
#     for B in meta_list:
#         if z < B[0]:
#             for i in range(3):
#                 B[i] = [x for j in range(i,p) if ]