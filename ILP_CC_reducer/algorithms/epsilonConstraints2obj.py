import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

# import sys
# import csv
# import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import algorithms_utils



class EpsilonConstraintAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, objectives_list: list) -> None:
        
        multiobj_model = MultiobjectiveILPmodel()
        
        obj1, obj2 = objectives_list[:2]
        
        output_data = []
        
        csv_data = [[f"{obj1.__name__}", f"{obj2.__name__}"]]
        
        # Define threshold
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold

        # Solve {min f2}
        multiobj_model.model.obj = pyo.Objective(rule=lambda m: obj2(m))
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
        
        result.write()
        
        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
            
            """ z <- Solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f2(z) := f2z
            f2z = pyo.value(concrete.obj)
            
            output_data.append("=====================================================================================================================================\n")
            if obj1.__name__ == 'LOCdifferenceObjective':
                valuef1 = concrete.tmax.value - concrete.tmin.value
            elif obj1.__name__ == 'CCdifferenceObjective':
                valuef1 = concrete.cmax.value - concrete.cmin.value
            elif obj1.__name__ == 'sequencesObjective':
                valuef1 = sum(concrete.x[i].value for i in concrete.S)
            # TODO: ME HE QUEDADO POR AQUÍ, SIGUE GENERALIZANDO TODO EL CÓDIGO    
            
            
            
            if obj2.__name__ == 'LOCdifferenceObjective':
                output_data.append(f"tmax: {concrete.tmax.value}, tmin: {concrete.tmin.value}\n")
            elif obj2.__name__ == 'CCdifferenceObjective':
                output_data.append(f"cmax: {concrete.cmax.value}, cmin: {concrete.cmin.value}\n")
            elif obj2.__name__ == 'sequencesObjective':
                output_data.append(f"sum: {[concrete.x[i].value for i in concrete.S]}\n")
            
            
            output_data.append(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}\n")
            
            
            
            
            
            
            
            output_data.append(f"Valores en el primer paso: {sum(concrete.x[i].value for i in concrete.S), valuef1}\n")
            
            # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobj_model.model.f2z = pyo.Param(initialize=f2z)
            # new constraint f2(x) <= f2(z)
            multiobj_model.model.f2Constraint = pyo.Constraint(rule=lambda m: multiobj_model.SecondObjdiffConstraint(m, obj2))
            # new objective: min f1(x)
            algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
            # Solve model
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
            # z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
            f1z = pyo.value(concrete.obj)
            
            output_data.append(f"min f1(x), sequences, subject to f2(x) <= f2(z): {f1z}\n")


            """ FP <- {z} (add z to Pareto front) """
            newrow = [sum(concrete.x[i].value for i in concrete.S), concrete.cmax.value - concrete.cmin.value] # Calculate results for CSV file
            csv_data.append(newrow)
            
            # algorithms_utils.write_results_and_sequences_to_file(concrete, f, result.solver.status, newrow, obj2)
            
            output_data.append('===============================================================================')
            if (result.solver.status == 'ok'):
                output_data.append(f'Objective SEQUENCES: {newrow[0]}')
                output_data.append(f'Objective {obj2}: {newrow[1]}')
                output_data.append('Sequences selected:')
                for s in concrete.S:
                    output_data.append(f"x[{s}] = {concrete.x[s].value}")
            output_data.append('===============================================================================')
            
            # epsilon <- f1(z) - 1
            multiobj_model.model.epsilon = pyo.Param(initialize=f1z-1, mutable=True)
            
            # lower bound for f1(x)
            u1 = f1z - 1
            
            # l = epsilon - f1(x)
            multiobj_model.model.l = pyo.Var(within=pyo.NonNegativeReals)

            
            solution_found = True # while loop control
            multiobj_model.model.del_component('f2Constraint') # delete f2(x) <= f2(z) constraint
            
            while solution_found:
                                
                # estimate a lambda value > 0
                lambd = 1/(f1z - u1)
                
                """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                # min f2(x) - lambda * l
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective_2obj(m, lambd, obj2)))
                # subject to f1(x) + l = epsilon
                algorithms_utils.modify_component(multiobj_model, 'epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint_2obj(m, 'SEQ')))
                # Solve
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                
                """ While exists x in X that makes f1(x) < epsilon do """
                if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                    
                    output_data.append(f"slack variable l value: {concrete.l.value}\n")
                    
                    # z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}
                    
                    """ PF = PF U {z} """
                    newrow = [sum(concrete.x[i].value for i in concrete.S), concrete.cmax.value - concrete.cmin.value] # Calculate results for CSV file
                    csv_data.append(newrow)
                    
                    
                    """ epsilon = f1(z) - 1 """
                    f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
                    
                    # New epsilon value
                    algorithms_utils.modify_component(multiobj_model, 'epsilon', pyo.Param(initialize=f1z-1, mutable=True))
                    
                    
                    output_data.append(f"f1z: {f1z}\n")
                    
                    
                    # lower bound for f1(x) (it has to decrease with f1z)
                    u1 = f1z - 1
                    
                    
                    
                    output_data.append(f"epsilon: {concrete.epsilon.value}\n")
                    output_data.append(f"u1: {u1}\n")
                    output_data.append(f"lambda: {lambd}\n")
                    output_data.append(f"comprobacion: {f1z} <= {concrete.epsilon.value}\n")
                    
                    # algorithms_utils.write_results_and_sequences_to_file(concrete, f, result.solver.status, newrow, obj2)
                    
                    output_data.append('===============================================================================')
                    if (result.solver.status == 'ok'):
                        output_data.append(f'Objective SEQUENCES: {newrow[0]}')
                        output_data.append(f'Objective {obj2}: {newrow[1]}')
                        output_data.append('Sequences selected:')
                        for s in concrete.S:
                            output_data.append(f"x[{s}] = {concrete.x[s].value}")
                    output_data.append('===============================================================================')
                    
                else:
                    solution_found = False            
            
        return csv_data, concrete, output_data


                



