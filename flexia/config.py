# Copyright 2022 The Flexia Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
        return Config(json.loads(string))
    
    def from_json(self, path:str) -> "Config":
        with open(path, "r", encoding="utf-8") as file:
            self = self.from_json_string(file.read())
        
        return self

    def __str__(self) -> str:
        attributes_string = ", ".join([f"{k}={v}" for k, v in self.items()])
        return f"Config({attributes_string})"

    __repr__ = __str__