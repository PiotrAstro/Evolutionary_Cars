

def get_policy_search_class(name: str):
    match name:
        case "Param_Les_Ev_Mut_Pop":
            from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Param_Less_Ev_Mut_Pop import Param_Les_Ev_Mut_Pop
            return Param_Les_Ev_Mut_Pop
        case "GESMR":
            from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.GESMR import GESMR
            return GESMR
        case "Evolutionary_Mutate_Population":
            from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Evolutionary_Mutate_Population import \
                Evolutionary_Mutate_Population
            return Evolutionary_Mutate_Population
        case "Evolutionary_Strategy":
            from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Evolutionary_Strategy import Evolutionary_Strategy
            return Evolutionary_Strategy
        case "Genetic_Algorithm":
            from src_files.Evolutionary_Algorithms.Genetic_Algorithm.Genetic_Algorithm import Genetic_Algorithm
            return Genetic_Algorithm
        case "Differential_Evolution":
            from src_files.Evolutionary_Algorithms.Genetic_Algorithm.Differential_Evolution import Differential_Evolution
            return Differential_Evolution
        case _:
            raise ValueError(f"Unknown policy search class name: {name}")