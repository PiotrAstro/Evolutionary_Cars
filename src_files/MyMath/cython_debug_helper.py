from typing import Dict, Any


def cython_debug_call(dict_args: Dict[str, Any], place_in_code: str = "none") -> None:
    """
    This function is used to debug cython code, you pass arguments as dict, and you can look at them here
    :param dict_args: dict [str, Any]
    :param place_in_code: str
    """

    print(f"cython_debug_call: {place_in_code}")
    print(dict_args)
    print("\n\n\n")