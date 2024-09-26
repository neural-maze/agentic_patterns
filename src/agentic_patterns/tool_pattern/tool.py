import json
from typing import Callable, Dict


def get_fn_signature(fn: Callable):
    fn_signature: Dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {
            "properties": {}
        }
    }
    schema = {k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"}
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


class Tool:
    def __init__(
        self,
        name: str, 
        fn: Callable,
        fn_signature: str
    ):  
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature
    
    def __str__(self):
        return self.fn_signature
    
    def run(self, **kwargs):
        return self.fn(**kwargs)


def tool(fn: Callable):
    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name"),
            fn=fn,
            fn_signature=json.dumps(fn_signature)
        )
    return wrapper()
