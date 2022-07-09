import json
from .third_party.addict import Dict


class Config(Dict):    
    def to_json_string(self) -> str:
        return json.dumps(self)
    
    def to_json(self, path:str) -> str:
        with open(path, "w", encoding="utf-8") as file:
            data = self.to_json_string()
            file.write(data)
        
        return path
    
    def from_json_string(self, string:str) -> dict:
        return self(json.loads(string))
    
    def from_json(self, path:str) -> "Config":
        with open(path, "r", encoding="utf-8") as file:
            self = self.from_json_string(file.read())
        
        return self

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.items()])
        return f"Config({attributes_string})"

    __repr__ = __str__