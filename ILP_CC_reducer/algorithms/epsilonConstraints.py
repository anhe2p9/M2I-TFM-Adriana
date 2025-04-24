import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

# from typing import Any
# import sys

import os
import csv

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import algorithms_utils



class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm with 3 obj'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int) -> None:
                
        multiobj_model = MultiobjectiveILPmodel()
        
        # f = open('output/eConstraint_output.txt', 'w')
        
        csv_data = [["Num.sequences","CCdif","LOCdif"]]        
        
        # Define threshold
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
        
        
        # Solve {min f2}
        multiobj_model.model.obj = pyo.Objective(rule=lambda m: multiobj_model.CCdifferenceObjective(m))
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
        
        
        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
            
            """ z <- Solve {min f3(x) subject to f2(x) <= f2(z)} """
            # f2(z) := f2z
            f2z = pyo.value(concrete.obj)
            
            print("=====================================================================================================================================")
            print(f"tmax: {concrete.tmax.value}, tmin: {concrete.tmin.value}")
            print(f"cmax: {concrete.cmax.value}, cmin: {concrete.cmin.value}")
            print(f"min f2(x), CC, subject to x in X: {f2z}")
            
            
            
            print(f"Valores en el primer paso: {algorithms_utils.calculate_results(concrete, 'CC')}")
            
            # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobj_model.model.f2z = pyo.Param(within=pyo.NonNegativeReals, initialize=f2z)
            # new constraint f2(x) <= f2(z)
            multiobj_model.model.f2Constraint = pyo.Constraint(rule=lambda m: multiobj_model.CCdifferenceObjective(m) <= m.f2z)
            
            
            
            # Solve {min f3}
            algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.LOCdifferenceObjective(m)))
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
            
            
            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                
                """ z <- Solve {min f1(x) subject to f2(x) <= f2(z), f3(x) <= f3(z)} """
                # f3(z) := f3z
                f3z = pyo.value(concrete.obj)
                
                
                # new parameter f3(z) to implement new constraint f3(x) <= f3(z)
                multiobj_model.model.f3z = pyo.Param(within=pyo.NonNegativeReals, initialize=f3z)
                # new constraint f3(x) <= f3(z)
                multiobj_model.model.f3Constraint = pyo.Constraint(rule=lambda m: multiobj_model.LOCdifferenceObjective(m) <= m.f3z)
                
                # new objective: min f1(x) (sequences)
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
                # Solve model
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                # z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
                f1z = pyo.value(concrete.obj)
                
                
                
                
                
                # new objective: min f1(x)
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
                # Solve model
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                # z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
                f1z = pyo.value(concrete.obj)
                
                print(f"min f1(x), sequences, subject to f2(x) <= f2(z): {f1z}")

    
                """ FP <- {z} (add z to Pareto front) """
                newrow = algorithms_utils.calculate_results(concrete, obj2) # Calculate results for CSV file
                csv_data.append(newrow)
                
                # algorithms_utils.print_result_and_sequences(result.solver.status, newrow, obj2, concrete)            
                
                # epsilon <- f1(z) - 1
                multiobj_model.model.epsilon = pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True)
                
                # lower bound for f1(x)
                u1 = f1z - 1
                
                # l2 = epsilon2 - f1(x)
                multiobj_model.model.l2 = pyo.Var(within=pyo.NonNegativeReals)
                multiobj_model.model.l3 = pyo.Var(within=pyo.NonNegativeReals)
    
                
                solution_found = True # while loop control
                multiobj_model.model.del_component('f2Constraint') # delete f2(x) <= f2(z) constraint
                
                while solution_found:
                                    
                    # estimate a lambda value > 0
                    lambd = 1/(f1z - u1)
                    
                    """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                    # min f2(x) - lambda * l
                    algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m, lambd, obj2)))
                    # subject to f1(x) + l = epsilon
                    algorithms_utils.modify_component(multiobj_model, 'epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ')))
                    # Solve
                    concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                    
                    """ While exists x in X that makes f1(x) < epsilon do """
                    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                        
                        print(f"slack variable l value: {concrete.l.value}")
                        
                        # z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}
                        
                        """ PF = PF U {z} """
                        newrow = algorithms_utils.calculate_results(concrete, obj2) # Calculate results for CSV file
                        csv_data.append(newrow)
                        
                        
                        """ epsilon = f1(z) - 1 """
                        f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
                        
                        # New epsilon value
                        algorithms_utils.modify_component(multiobj_model, 'epsilon', pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True))
                        
                        
                        print(f"f1z: {f1z}")
                        
                        
                        # lower bound for f1(x) (it has to decrease with f1z)
                        u1 = f1z - 1
                        
                        
                        
                        print(f"epsilon: {concrete.epsilon.value}")
                        print(f"u1: {u1}")
                        print(f"lambda: {lambd}")
                        print(f"comprobacion: {f1z} <= {concrete.epsilon.value}")
                        
                        # algorithms_utils.print_result_and_sequences(result.solver.status, newrow, obj2, concrete)
                        
                    else:
                        solution_found = False
                
                # f.close()
                
                # Save model in a LP file
                concrete.write(f'output/Econstraint_FALTA_PONER_EL_NOMBRE_DEL_METODO.lp', io_options={'symbolic_solver_labels': True})
                
                # Write data in a CSV file.
                write_file(csv_data)


                



def write_file(csv_info: list):
    filename = "output/epsilon_constr_2obj_output.csv"
            
    if os.path.exists(filename):
        os.remove(filename)
            
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_info)
        print("CSV file correctly created.")