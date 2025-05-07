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
    def execute(data: dp.DataPortal, tau: int):
                
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
                
                
                
                # new objective: min f1(x)
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
                # Solve model
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                # z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
                f1z = pyo.value(concrete.obj)
                
                print(f"min f1(x), sequences, subject to f2(x) <= f2(z), f3(x) <= f3(z): {f1z}")

    
                """ FP <- {z} (add z to Pareto front) """
                newrow = algorithms_utils.calculate_results(concrete) # Calculate results for CSV file
                csv_data.append(newrow)
                
                algorithms_utils.print_result_and_sequences(concrete, result.solver.status, newrow)            
                
                # epsilon2 <- f1(z) - 1
                multiobj_model.model.epsilon1 = pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True)
                # epsilon2 <- f2(z) - 1
                multiobj_model.model.epsilon2 = pyo.Param(within=pyo.NonNegativeReals, initialize=f2z-1, mutable=True)
                
                
                # lower bound for f1(x)
                u1 = f1z - 1
                # lower bound for f2(x)
                u2 = f2z - 1
                
                # l1 = epsilon1 - f1(x)
                multiobj_model.model.l1 = pyo.Var(within=pyo.NonNegativeReals)
                # l2 = epsilon2 - f2(x)
                multiobj_model.model.l2 = pyo.Var(within=pyo.NonNegativeReals)
    
                
                solution_found = True # while loop control
                multiobj_model.model.del_component('f2Constraint') # delete f2(x) <= f2(z) constraint
                multiobj_model.model.del_component('f3Constraint') # delete f3(x) <= f3(z) constraint
                
                while solution_found:
                
                    # estimate a lambda1 value > 0
                    multiobj_model.model.lambd1 = pyo.Param(within=pyo.NonNegativeReals, initialize=1/(f1z - u1))
                    # estimate a lambda2 value > 0
                    multiobj_model.model.lambd2 = pyo.Param(within=pyo.NonNegativeReals, initialize=1/(f2z - u2))
                    
                    """ Solve {min f3(x) - (lambda1 * l1 + lambda2 * l2), subject to f1(x) + l1 = epsilon1, f2(x) + l2 = epsilon2} """
                    # min f2(x) - lambda * l
                    algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m)))
                    
                    
                    # subject to f1(x) + l1 = epsilon1
                    algorithms_utils.modify_component(multiobj_model, 'epsilonConstraint1', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ')))
                    # subject to f2(x) + l2 = epsilon2
                    algorithms_utils.modify_component(multiobj_model, 'epsilonConstraint2', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'CC')))
                    
                    
                    # Solve
                    concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                    
                    """ While exists x in X that makes f1(x) < epsilon do """
                    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                        
                        print(f"slack variable l value: {concrete.l.value}")
                        
                        # z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}
                        
                        """ PF = PF U {z} """
                        newrow = algorithms_utils.calculate_results(concrete) # Calculate results for CSV file
                        csv_data.append(newrow)
                        
                        
                        """ epsilon1 = f1(z) - 1 """
                        f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
                        # New epsilon1 value
                        algorithms_utils.modify_component(multiobj_model, 'epsilon1', pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True))
                        
                        
                        """ epsilon2 = f2(z) - 1 """
                        f2z = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
                        # New epsilon2 value
                        algorithms_utils.modify_component(multiobj_model, 'epsilon2', pyo.Param(within=pyo.NonNegativeReals, initialize=f2z-1, mutable=True))
                        
                        
                        print(f"f1z: {f1z}")
                        print(f"f1z: {f2z}")
                        
                        
                        # lower bound for f1(x) (it has to decrease with f1z)
                        u1 = f1z - 1
                        # lower bound for f2(x) (it has to decrease with f2z)
                        u2 = f2z - 1
                        
                        
                        
                        print(f"epsilon1: {concrete.epsilon1.value}")
                        print(f"epsilon2: {concrete.epsilon2.value}")
                        print(f"u1: {u1}")
                        print(f"u2: {u2}")
                        print(f"lambda1: {concrete.lambd1.value}")
                        print(f"lambda2: {concrete.lambd2.value}")
                        print(f"comprobacion1: {f1z} <= {concrete.epsilon1.value}")
                        print(f"comprobacion2: {f2z} <= {concrete.epsilon2.value}")
                        
                        algorithms_utils.print_result_and_sequences(concrete, result.solver.status, newrow)
                        
                    else:
                        solution_found = False
                
                # f.close()
                
                return csv_data, concrete