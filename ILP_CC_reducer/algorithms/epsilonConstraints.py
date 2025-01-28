



class EpsilonConstraintAlgorithm(FMRefactoring):

    @staticmethod
    def get_name() -> str:
        return 'Excludes constraint refactoring'
    
    @staticmethod
    def get_description() -> str:
        return ("It translate an excludes constraint from the feature model to the feature tree.")

    @staticmethod
    def get_language_construct_name() -> str:
        return 'Excludes constraint'

    @staticmethod
    def get_instances(model: FeatureModel) -> list[Constraint]:
        return [ctc for ctc in model.get_constraints() 
                if constraints_utils.is_excludes_constraint(ctc)]

    @staticmethod
    def is_applicable(model: FeatureModel) -> bool:
        return len(ExcludesConstraintRefactoring.get_instances(model)) > 0

    @staticmethod
    def transform(model: FeatureModel, instance: Constraint) -> FeatureModel:
        return eliminate_excludes(model, instance)
