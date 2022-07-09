import numpy as np
from copy import deepcopy
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from typing import Any, Optional, Union, Tuple
from .utils import create_probability_mask
from .constants import supported_dtypes


def apply_masking(inputs:Any, 
                  tokenizer:PreTrainedTokenizer, 
                  special_token_ids:Optional[Any]=None, 
                  mlm_probability:float=0.15, 
                  return_labels:bool=True, 
                  ignore_index:Optional[int]=-100, 
                  mask_token_probability:float=0.8, 
                  mask_token_id:Optional[int]=None,
                  random_token_probability:float=0.1, 
                  dtype:str="pt",
                  debug=False) -> Union[Any, Tuple[Any, Any]]:
    
    """
    Implementation of Masked Language Modeling - https://arxiv.org/abs/1810.04805
    
    
    Inputs:
        inputs: Any - input for masking.
        tokenizer: PreTrainedTokenizer - tokenizer from HuggingFace Transformers library.
        special_token_ids: Optional[Any] - special token indices mask to prevent selecting important tokens, e.g [CLS], [PAD], [SEP]. Default: None.
        mlm_probability: float - fraction/probability of tokens (selected tokens), which will be masked. Default: 0.15.
        return_labels: bool - function returns labels, if True. Default: True.
        ignore_index: Optional[int] - specify index for loss, in order to compute the loss on only masked tokens. Default: -100.
        mask_token_probability: float - fraction/probability of tokens from selected tokens, which will be masked with [MASK] token. Default: 0.8.
        random_token_probability: float - fraction/probability of tokens from selected tokens, which will be masked with random token from tokenizer's vocabulary. Default: 0.1.
        debug: bool - debug the masking operation. Default: False.
        dtype: str - data type of returning masked inputs and labels, must be on of ["pt", "np", "list"]. Default: "pt".
    
    Returns:
        inputs: Any - masked inputs.
        labels: Any - labels, i.e original inputs. It will be returned if `return_labels` is True.
    
    """
    
    unchanged_probability = 1 - (mask_token_probability + random_token_probability)
    overall_probability = (mask_token_probability + random_token_probability + unchanged_probability)
    
    if not (0 <= mlm_probability <= 1):
        raise ValueError(f"`mlm_probability` must be in range [0, 1], but given {mlm_probability}.")
    
    if overall_probability != 1:
        raise ValueError(f"Overall sum of probabilities must be in range [0, 1], but given {overall_probability}.")
        
    if dtype not in supported_dtypes:
        raise ValueError(f"The given `dtype` {dtype} is not provided. Choose one of {supported_dtypes}.")
    
    dtype = supported_dtypes[dtype]
    
    inputs = deepcopy(inputs)
    
    if isinstance(inputs, BatchEncoding):
        special_token_ids = inputs.get("token_type_ids")
        inputs = inputs.input_ids
    
    if special_token_ids is None:
        try:
            special_token_ids = [tokenizer.get_special_tokens_mask(input, already_has_special_tokens=True) for input in inputs]
        except TypeError:
            special_token_ids = tokenizer.get_special_tokens_mask(inputs, already_has_special_tokens=True)
            
        special_token_ids = np.array(special_token_ids).astype(bool)
    
    inputs = np.asarray(inputs)
    labels = deepcopy(inputs)

    mask_indexes = create_probability_mask(inputs, probability=mlm_probability)
    mask_indexes = mask_indexes & ~special_token_ids if special_token_ids is not None else mask_indexes
        
    if ignore_index is not None:
        labels[~mask_indexes] = ignore_index
    
    mask_token_indexes = create_probability_mask(inputs, probability=mask_token_probability) & mask_indexes
    inputs[mask_token_indexes] = tokenizer.mask_token_id if mask_token_id is None else mask_token_id
    
    try:
        random_token_probability = random_token_probability * (1 / (1 - mask_token_probability))
    except ZeroDivisionError:
        random_token_probability = 0
        
    random_token_indexes = create_probability_mask(inputs, probability=random_token_probability) & mask_indexes & ~mask_token_indexes
    random_tokens = np.random.randint(low=0, high=tokenizer.vocab_size, size=inputs.shape)
    inputs[random_token_indexes] = random_tokens[random_token_indexes]
    
    inputs, labels = inputs.squeeze(), labels.squeeze()
    inputs, labels = dtype(inputs), dtype(labels)
    
    if debug:
        num_masked_tokens = np.sum(mask_indexes)
        num_mask_tokens = np.sum(mask_token_indexes)
        num_random_tokens = np.sum(random_token_indexes)
        num_unchanged_tokens = num_masked_tokens - (num_mask_tokens + num_random_tokens)
        
        print(f"Masked tokens: {num_masked_tokens}")
        print(f"[MASK] tokens: {num_mask_tokens}")
        print(f"Random tokens: {num_random_tokens}")
        print(f"Unchanged tokens: {num_unchanged_tokens}")
    
    return (inputs, labels) if return_labels else mask_indexes