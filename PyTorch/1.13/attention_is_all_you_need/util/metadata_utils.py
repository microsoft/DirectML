"""
Utils for model metadata
"""

from typing import Any, List, Dict

def match_item(item_name: str, item_val: str, skip_item: Dict[str, Any]) -> bool:
    if item_name not in skip_item:
        return True
    return skip_item[item_name] == item_val

def skip_by_metadata(test: str, device:str, jit: bool, extra_args: List[str], metadata: Dict[str, Any]) -> bool:
    "Check if the test should be skipped based on model metadata."
    if not "not_implemented" in metadata:
        return False
    for skip_item in metadata["not_implemented"]:
        match = match_item("test", test, skip_item) and \
                match_item("device", device, skip_item) and \
                match_item("jit", jit, skip_item) and \
                match_item("extra_args", extra_args, skip_item)
        if match:
            return True
    return False