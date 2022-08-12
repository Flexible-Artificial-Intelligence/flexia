from typing import List, Any

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


def pad_sequence(sequence:List[int], 
                 max_length:int, 
                 padding_value:Any=-1, 
                 padding_side="right"
                 ) -> List[int]:
    
    sequence_length = len(sequence)
    length_diff = max_length - sequence_length

    padding_value = [padding_value]
    padding_values = padding_value * length_diff

    if padding_side == "left":
        return padding_values + sequence
    else:
        return sequence + padding_values


def pad_sequences(sequences:List[List[int]], **kwargs) -> List[List[int]]:
    return [pad_sequence(sequence=sequence, **kwargs) for sequence in sequences]


# TO-DO
def convert_tokens_to_words_predictions():
    pass

def convert_words_to_chars_predictions():
    pass

def convert_words_to_tokens_predictions():
    pass

def convert_chars_to_tokens_predictions():
    pass