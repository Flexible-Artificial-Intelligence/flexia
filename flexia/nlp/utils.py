from typing import List

from ..import_utils import is_transformers_available


if is_transformers_available():
    from transformers import PreTrainedTokenizer


def convert_ids_to_string(ids:List[int], 
                          tokenizer:PreTrainedTokenizer, 
                          skip_special_tokens:bool=False
                          ) -> str:
    
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
    string = tokenizer.convert_tokens_to_string(tokens)

    return string