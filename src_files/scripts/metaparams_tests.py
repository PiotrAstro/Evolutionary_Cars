import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from src_files.Evolutionary_Algorithms.Evolutionary_Strategies.Evolutionary_Mutate_Population import Evolutionary_Mutate_Population
from src_files.constants import CONSTANTS_DICT


# run this script from terminal, be in directory Evolutionary_Cars and paste:
# python -m src_files.scripts.metaparams_tests

POLICY_SEARCH_ALGORITHM = Evolutionary_Mutate_Population
TESTS_TRIES = 6
CONCURRENT_WORKERS = 5
SAVE_DIR = r"logs/metaparameters_tests_" + str(int(time.time()))


# everything should be in list - so it can be iterated, base level of dictionary should be list
# examples:
# TESTED_VALUES = [
#     {
#         "Evolutionary_Mutate_Population": {
#             "mutation_factor": [0.01, 0.05, 0.1, 0.2],
#             "mutation_threshold": [None],
#         },
#     },
#
#     {
#         "Evolutionary_Mutate_Population": {
#             "mutation_factor": [0.03, 0.1, 0.3],
#             "mutation_threshold": [0.03, 0.1, 0.3],
#         },
#     }
# ]
#
# or
# TESTED_VALUES = [{
#         "Evolutionary_Mutate_Population": {
#             "mutation_factor": [0.01, 0.05, 0.1, 0.2],
#             "mutation_threshold": [None],
#         },
#     }]
#
# or
# TESTED_VALUES = [{
#         "mutation_factor": [0.01, 0.05, 0.1, 0.2],
#         "mutation_threshold": [None],
#         "some_list_argument": [[1, 2], [8, 9]],
#     }]


TESTED_VALUES = [
    {
        "Evolutionary_Mutate_Population": {
            "mutation_factor": [0.05],
            "L1": [0.0, 0.0001],
            "L2": [0.0, 0.0001],
            "max_threads": [5],
            "use_safe_mutation": [False],
        },
    },

    # {
    #     "Evolutionary_Mutate_Population": {
    #         "mutation_factor": [0.002, 0.001],
    #         "use_safe_mutation": [True],
    #     },
    # }
]


def create_all_special_dicts(tested_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates all special dictionaries to test from given tested_values scheme
    :param tested_values:
    :return:
    """
    def find_lists_in_dict(dict_to_search: Dict[str, Any]) -> List[Tuple[List[str], List[Any]]]:
        """
        Finds all lists in dictionary
        :param dict_to_search:
        :return:
        """
        lists_keys_values = []
        for key, value in dict_to_search.items():
            if isinstance(value, list):
                lists_keys_values.append(
                    ([key], value)
                )
            elif isinstance(value, dict):
                tmp_lists_keys_values = find_lists_in_dict(value)
                for key_list, value_list in tmp_lists_keys_values:
                    lists_keys_values.append(([key] + key_list, value_list))
        return lists_keys_values

    def construct_special_dicts(keys_values_to_add: List[Tuple[List[str], List[Any]]], so_far_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Constructs special dictionary from given keys_values_to_add
        :param keys_values_to_add:
        :param so_far_dict:
        :return:
        """
        special_dicts = []
        if len(keys_values_to_add) == 0:
            special_dicts.append(so_far_dict)
        else:
            key_list, value_list = keys_values_to_add[0]
            for value in value_list:
                new_dict = copy.deepcopy(so_far_dict)
                tmp_dict = new_dict
                for key in key_list[:-1]:
                    if key not in tmp_dict:
                        tmp_dict[key] = {}
                    tmp_dict = tmp_dict[key]
                tmp_dict[key_list[-1]] = value
                special_dicts += construct_special_dicts(keys_values_to_add[1:], new_dict)
        return special_dicts

    special_dicts = []
    for tested_value in tested_values:
        lists_keys_values_tmp = find_lists_in_dict(tested_value)
        special_dicts += construct_special_dicts(lists_keys_values_tmp, {})

    return special_dicts


def test_one_case(basic_dict: Dict[str, Any], special_dict: Dict[str, Any], case_index: int) -> None:
    def change_dict_value(destination_dict: Dict[str, Any], source_dict: Dict[str, Any]) -> None:
        """
        Changes values in destination_dict to values from source_dict, all source_dict keys must be in destination_dict
        :param destination_dict:
        :param source_dict:
        :return:
        """
        for key, value in source_dict.items():
            if isinstance(value, dict):
                change_dict_value(destination_dict[key], value)
            else:
                destination_dict[key] = value

    def dict_to_name(dict_to_use: Dict[str, Any]) -> str:
        """
        Converts dictionary to string
        :param dict_to_use:
        :return:
        """
        name = "("
        for key, value in dict_to_use.items():
            if isinstance(value, dict):
                name += shorten_name(key) + "-" + dict_to_name(value)
            else:
                name += shorten_name(key) + "-" + str(value)
            name += "_"
        name = name[:-1] + ")"
        return name

    def shorten_name(name: str, cut_length: int = 3) -> str:
        """
        Shortens name of variable from constants_dict
        :param name:
        :param cut_length: minimal length of word, >= 1
        :return:
        """
        words = name.split("_")
        for k in range(len(words)):
            if len(words[k]) > cut_length:
                words[k] = words[k][:cut_length]
            words[k] = words[k].capitalize()
        return "".join(words)

    dict_copy = copy.deepcopy(basic_dict)
    change_dict_value(dict_copy, special_dict)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_name = SAVE_DIR + r"/logs_" + dict_to_name(special_dict) + "__case" + str(case_index) + ".csv"

    searching_method = POLICY_SEARCH_ALGORITHM(dict_copy)
    logs_df = searching_method.run()

    logs_df.to_csv(save_name, index=False, sep=",")


if __name__ == "__main__":
    possible_dicts = create_all_special_dicts(TESTED_VALUES)

    futures = []
    with ProcessPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        # Submit all the tasks to the executor
        for special_dict in possible_dicts:
            for i in range(TESTS_TRIES):
                # Submitting the task to be executed in parallel
                future = executor.submit(test_one_case, CONSTANTS_DICT, special_dict, i)
                futures.append(future)

        # Process the results as they are completed
        for future in as_completed(futures):
            result = future.result()
            print(result)

# Specify the number of concurrent workers
concurrent_workers = 4  # This can be adjusted based on your machine's capabilities

