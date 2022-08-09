"""
References
    https://natural-language-understanding.fandom.com/wiki/Named_entity_recognition
    https://www.sciencedirect.com/science/article/pii/S1110866520301596
"""

import numpy as np
from typing import List, Tuple, Optional, Any


# TAGGING
def generate_bio_tagging(entities:List[str], 
                         spans:List[List[int]], 
                         offset_mapping:List[Tuple[int, int]],
                         ) -> List[str]:
    """
    BIO (Beginning-Inside-Outside) tagging

    References:
        https://github.com/affjljoo3581/Feedback-Prize-Competition

    """

    num_tokens = len(offset_mapping)
    tags = ["O"]*num_tokens
    for entity, (entity_start, entity_end) in zip(entities, spans):
        current_tag = f"B-{entity}"
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            if min(entity_end, token_end) - max(entity_start, token_start) > 0:
                tags[token_index] = current_tag
                current_tag = f"I-{entity}"
                
    return tags


# TO-DO
def generate_io_tagging(entities:List[str], 
                        spans:List[List[int]], 
                        offset_mapping:List[Tuple[int, int]],
                        ) -> List[str]:
    """
    IO (Inside-Outside) tagging
    """

    pass

def generate_eio_tagging(entities:List[str], 
                         spans:List[List[int]], 
                         offset_mapping:List[Tuple[int, int]],
                         ) -> List[str]:
    """
    EIO (Ending-Inside-Outside) tagging
    """

    pass


def generate_bieo_tagging(entities:List[str], 
                          spans:List[List[int]], 
                          offset_mapping:List[Tuple[int, int]],
                          ) -> List[str]:
    """
    BIEO (Beginning-Inside-Ending-Outside) tagging
    """

    pass

def generate_bieso_tagging(entities:List[str], 
                           spans:List[List[int]], 
                           offset_mapping:List[Tuple[int, int]],
                           ) -> List[str]:
    """
    BIESO (Beginning-Inside-Ending-Single-Outside)
    """

    pass

def generate_bilou_tagging(entities:List[str], 
                           spans:List[List[int]], 
                           offset_mapping:List[Tuple[int, int]],
                           ) -> List[str]:
    """
    BILOU (Beggining-Inside-Last-Outside-Unit)
    """

    pass


# Aliases
generate_iob_tagging = generate_bio_tagging
generate_ioe_tagging = generate_eio_tagging
generate_iobe_tagging = generate_bieo_tagging
generate_iobes_tagging = generate_bieso_tagging
generate_ioblu_tagging = generate_bilou_tagging





# AUGMENTATIONS
def cutmix(input_ids:List[List[int]], 
           attention_mask:List[List[int]], 
           target:List[List[Any]], 
           p:float=0.5, 
           cut:float=0.25
           ) -> Tuple[List[List[int]], List[List[int]], List[List[Any]]]:

    if np.random.uniform() < p:
        batch_size, length = input_ids.size()
            
        permutation = np.random.permutation(batch_size)
        random_length = int(length*cut)
        start = np.random.randint(length-random_length)
        input_ids[:,start:start+random_length] = input_ids[permutation,start:start+random_length]
        attention_mask[:,start:start+random_length] = attention_mask[permutation,start:start+random_length]
        target[:,start:start+random_length] = target[permutation,start:start+random_length]
        
    return input_ids, attention_mask, target