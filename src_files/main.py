import math
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Differential_Evolution import Differential_Evolution
from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Evolutionary_Mutate_Population import \
    Evolutionary_Mutate_Population
from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Evolutionary_Strategy import Evolutionary_Strategy
from src_files.Evolutionary_Algorithms.Genetic_Algorithm.Genetic_Algorithm import Genetic_Algorithm
from src_files.constants import CONSTANTS_DICT

# run this script from terminal, be in directory Stock_Agent and paste:
# python -m src_files.main.py

# spróbować SHADE Differential Evolution, uważać na upiększanie - oszukano na dekompozycji
# Omidvar - dekompozycja w przestrzeniach ciągłych
# Xiaodong Li - też,
# napisać maila o tym!
# IRRG - Komarnicki
# sprawdzić iterated local search

# moje pomysły - l1 i l2
# wtedy sprawdzić jeszcze raz z różnymi wartościami mutacji w zależności od wartości bezwzględnej parametru
# dynamiczna zmiana populacji (jak idzie dobrze to mniejsza, jak źle to większa)
# mutacje - dziecko dziedziczy mutacje po rodzicu, początkowo losuję z normal distribution mutacje, co np 10 generacji biorę średni poziom mutacji i losuję z niego z normal distribution


# policy_search_algorithm = Differential_Evolution(CONSTANTS_DICT)
policy_search_algorithm = Evolutionary_Mutate_Population(CONSTANTS_DICT)
# policy_search_algorithm = Evolutionary_Strategy(CONSTANTS_DICT)
# policy_search_algorithm = Genetic_Algorithm(CONSTANTS_DICT)
policy_search_algorithm.run()

# run_basic_environment_visualization()
