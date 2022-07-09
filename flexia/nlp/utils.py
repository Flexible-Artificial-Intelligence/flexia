from transformers import PreTrainedTokenizer
from typing import Any, Tuple, Union, Optional
import numpy as np

from ..utils import unsqueeze


def convert_ids_to_string(ids:Any, tokenizer:PreTrainedTokenizer) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    string = tokenizer.convert_tokens_to_string(tokens)
    
    return string


def create_probability_mask(inputs:Any, probability:float=0.5) -> np.ndarray:
    probability_matrix = np.random.uniform(low=0, high=1, size=inputs.shape)
    return (probability_matrix < probability).astype(bool)


def get_special_tokens_indexes(tokenized):
    sequence_ids = np.array(tokenized.sequence_ids())
    special_tokens_indexes = np.where(sequence_ids == None)[0]
    return special_tokens_indexes


def avoid_special_tokens(offset_mapping:list) -> list:
    """
    Removes special tokens ((0, 0)) from offset_mapping.
    
    Inputs:
        offset_mapping: list - offset_mapping from HuggingFace Transformers tokenizer.
    
    Returns:
        offset_mapping: list - offset_mapping without special tokens.
        
    Examples:
    
        >>> text = "Single example" 
        >>> tokenized = tokenizer(text, max_length=7, padding="max_length", return_offsets_mapping=True)
        >>> offset_mapping = tokenized["offset_mapping"]
        >>> offset_mapping
        >>> [(0, 0), (0, 4), (4, 11), (11, 19), (0, 0), (0, 0), (0, 0)]
        >>> new_offset_mapping = avoid_special_tokens(offset_mapping)
        >>> new_offset_mapping
        >>> [(0, 4), (4, 11), (11, 19)]
        
    """
    
    offset_mapping = [(start, end) for (start, end) in offset_mapping if (start != 0) or (end != 0)]
    return offset_mapping



def compare_text_with_offset_mapping(text:str, offset_mapping:list, ignore_special_tokens:bool=False) -> list:
    """
    Compares 'text' with 'offset_mapping' from HuggingFace Transformers tokenizers.
    
    Inputs:
        text: str - input text.
        offset_mapping: list - offset_mapping from HuggingFace Transformers tokenizer.
        ignore_special_tokens: bool - ignores special tokens in 'offset_mapping'. Default: False.
        
    Returns:
        comparing: list - results of comparing.
        
    Examples:
    
        >>> text = "Single example" 
        >>> tokenized = tokenizer(text, max_length=7, padding="max_length", return_offsets_mapping=True)
        >>> offset_mapping = tokenized["offset_mapping"]
        >>> offset_mapping
        >>> [(0, 0), (0, 6), (6, 14), (0, 0), (0, 0), (0, 0), (0, 0)]
        >>> comparing_without_special_tokens = compare_text_with_offset_mapping(text=text, offset_mapping=offset_mapping, ignore_special_tokens=True)
        >>> comparing_without_special_tokens
        >>> [[(0, 6), 'Single'], [(6, 14), ' example']]
        >>> comparing_with_special_tokens = compare_text_with_offset_mapping(text=text, offset_mapping=offset_mapping, ignore_special_tokens=False)
        >>> comparing_with_special_tokens
        >>> [[(0, 0), ''], [(0, 6), 'Single'], [(6, 14), ' example'], [(0, 0), ''], [(0, 0), ''], [(0, 0), ''], [(0, 0), '']]
    
    """
    
    offset_mapping = avoid_special_tokens(offset_mapping) if ignore_special_tokens else offset_mapping
    
    comparing = []
    for (start, end) in offset_mapping:
        span = (start, end)
        substring = text[start:end]
        comparing.append([span, substring])
            
    return comparing



def cutmix(input_ids:Any, 
           attention_mask:Any, 
           target:Optional[Any]=None, 
           p:float=0.25, 
           cut:float=0.1) -> Union[Tuple[Any, Any, Any], Tuple[Any, Any]]:
    """
    Implementation of CutMix augmentation for the text.
    Inputs:
        input_ids: Any - input ids from HuggingFace Transformers tokenizer. Shape: (batch size, max length)
        attention_mask: Any - attention mask for input ids from HuggingFace Transformers tokenizer. Shape: (batch size, max length)
        target: Any - target for the input ids. Shape (batch size, max length, num classes)
        p: float - probability of applying CutMix. Must be in range (0, 1]. Default: 0.25.
        cut: float - percent of sequence, which will be cutted. Must be in range [0, 1). Default: 0.1.
    Returns:
        input_ids: np.array - transformed input_ids.
        attention_mask: np.array - transformed attention_mask.
        target: np.array - transformed target.
    Examples:
        ---
    """

    if not (0 < p <= 1):
        raise ValueError(f"Probability parameter 'p' must be in range (0, 1].")

    if not (0 <= cut < 1):
        raise ValueError(f"Cut parameter 'cut' must be in range [0, 1).")


    if np.random.uniform() < p:
        if len(input_ids.shape) == 1:
            input_ids = unsqueeze(input_ids, dim=0)
            attention_mask = unsqueeze(attention_mask, dim=0)

        batch_size, length, *_ = input_ids.shape
        permutation = np.random.permutation(batch_size)
        random_length = int(length*cut)
        start = np.random.randint(length-random_length)
        end = start + random_length

        input_ids[:,start:end] = input_ids[permutation,start:end]
        attention_mask[:,start:end] = attention_mask[permutation,start:end]

        if target is not None:
            if len(target.shape) == 1:
                target = unsqueeze(target, dim=0)

            target[:,start:end] = target[permutation,start:end]

    if target is not None:
        return input_ids, attention_mask, target
    
    return input_ids, attention_mask


def get_missing_spans(spans:Any, difference:int=1) -> list:
    missing_spans = []
    for span, next_span in zip(spans, spans[1::]):
        (start, end), (next_start, next_end) = span, next_span
        
        if not ((next_start - end) <= difference):
            missing_span = [end+1, next_start]
            missing_spans.append(missing_span)
            
    return missing_spans


def filter_spans(spans:Any) -> np.ndarray:
    spans = np.array(spans)
    starts, ends = spans[:, 0], spans[:, 1]
    sorted_starts = np.argsort(starts)
    spans = spans[sorted_starts]
    return spans


def concatenate_spans(x_spans:Any, y_spans:Any) -> np.ndarray:
    spans = np.concatenate([x_spans, y_spans], axis=0)
    spans = filter_spans(spans)
    return spans