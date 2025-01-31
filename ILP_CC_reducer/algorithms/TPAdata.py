import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from pyomo.core.expr.visitor import identify_variables

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class TPAdataAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'TPA data Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains model matrices to apply TPA Algorithm.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal) -> None:
        
        concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
        
        def get_matrix_vector(concrete):
            A = []  # Matriz de coeficientes de las restricciones
            b = []  # Vector de términos independientes
            c = []  # Vector de la función de coste
        
            # Obtener coeficientes de las restricciones
            
            # for i in range(3):
            #     print(f"Coeficiente de x[{i}]: {concrete.constraint.expr.get(i, 0)}")
            # print(f"Coeficiente de y[0]: {concrete.constraint.expr.get(model.y[0], 0)}")
            
            for constraint in concrete.component_data_objects(pyo.Constraint):
                row = []
                print(f"CONSTRAINT: {constraint}")
                print(f"CONSTRAINT INDEX: {constraint.index()}")
                print(f"CONSTRAINT BODY: {constraint.body}")
                
                if isinstance(constraint.index(), int):
                    for term in constraint.body.args:
                        if isinstance(term, tuple):  # Este es un término que tiene coeficiente y variable
                            coef = term[0]
                            var = term[1]
                            print(f"Variable: {var}, Coeficiente: {coef}")
                        else:
                            print(f"Termino: {term}")
                            
                            
                            
                    
                print(f"COEFFICIENTS: {constraint.body.linear_coefs[0]}")
                
                
                for var in concrete.component_data_objects(pyo.Var):
                    coef = 0
                    # Iterar sobre los términos de la expresión lineal en la restricción
                    
                    # TODO
                    
                    
                    
                    
                    
                    
                    row.append(coef)
                A.append(row)
                b.append(constraint.upper())
        
            # Obtener coeficientes de la función objetivo
            for var in concrete.component_data_objects(pyo.Var):
                c.append(concrete.obj.get_coeff(var))
        
            return A, b, c
        
        # Llamar a la función
        A, b, c = get_matrix_vector(concrete)
        
        # Imprimir resultados
        print("Matriz de coeficientes de las restricciones:", A)
        print("Vector de términos independientes:", b)
        print("Vector de la función de costes:", c)
