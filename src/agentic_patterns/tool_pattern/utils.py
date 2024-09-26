import json

def validate_arguments(tool_call: dict, tool_signature: dict):
    """
    Validates if the arguments in the input dictionary match the types specified in the schema.
    """
    properties = tool_signature["parameters"]["properties"]
    
    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }
    
    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")

        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)
            
    return tool_call