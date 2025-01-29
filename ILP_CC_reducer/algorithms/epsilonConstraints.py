import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer




class EpsilonConstraintAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: pyo.ConcreteModel, w1: float, w2: float, w3: float) -> None:
        pass
