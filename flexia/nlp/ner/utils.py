"""
References
    https://natural-language-understanding.fandom.com/wiki/Named_entity_recognition
    https://www.sciencedirect.com/science/article/pii/S1110866520301596
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Union, Dict


# TAGGING
def generate_entity2id(entities:List[str], 
                       inside_token:str="I-",
                       outside_token:str="O", 
                       beginning_token:Optional[str]=None, 
                       ending_token:Optional[str]=None,
                       single_token:Optional[str]=None,
                       sort:bool=False, 
                       return_id2entity:bool=False
                       ) -> Union[Dict[str, int], Tuple[Dict[str, int], Dict[int, str]]]:
    
    if sort:
        entities = sorted(entities)
    
    entity2id = {}
    entity_format = "{token}{entity}"

    # Outside token
    entity2id[outside_token] = 0

    index = 1
    for entity in entities:
        # Beginning token
        if beginning_token is not None:
            beginning_entity = entity_format.format(token=beginning_token, entity=entity)
            entity2id[beginning_entity] = index
            index += 1
        
        # Inside token
        inside_entity = entity_format.format(token=inside_token, entity=entity)
        entity2id[inside_entity] = index
        index += 1

        # Ending token
        if ending_token is not None:
            ending_entity = entity_format.format(token=ending_token, entity=entity)
            entity2id[ending_entity] = index
            index += 1

        # Single token
        if single_token is not None:
            single_entity = entity_format.format(token=single_entity, entity=entity)
            entity2id[single_entity] = index
            index += 1
    
    # Converting entity2id to id2entity
    if return_id2entity:
        id2entity = {id_: label for label, id_ in entity2id.items()}
        return entity2id, id2entity
    
    return entity2id


generate_bio_entity2id = lambda *args, **kwargs: generate_entity2id(beginning_token="B-", 
                                                                    *args, 
                                                                    **kwargs)



def convert_entities_to_ids(entities, entity2id):
    return [entity2id[entity] for entity in entities]


def convert_ids_to_entities(ids, id2entity):
    return [id2entity[id_] for id_ in ids]


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


def filter_entities(entities:List[str], 
                    spans:List[List[int]], 
                    probabilities:List[Any], 
                    min_lengths:Dict[str, int]={}, 
                    min_probabilities:Dict[str, float]={},
                    strict=False,
                    ) -> Tuple[List[str], List[List[int]], List[Any]]:

    def compare(a, b, strict=False):        
        return a > b if strict else a >= b

    filtered_entities, filtered_spans, filtered_probabilities = [], [], []
    for entity, span, probability in zip(entities, spans, probabilities):
        start, end = span
        entity_length = end - start + 1

        length_condition = True
        if len(min_lengths) > 0:
            min_entity_length = min_lengths[entity]
            length_condition = compare(entity_length, min_entity_length, strict=strict)
        
        probability_condition = True
        if len(min_probabilities) > 0:
            min_entity_probability = min_probabilities[entity]
            probability_condition = compare(probability, min_entity_probability, strict=strict)

        if length_condition and probability_condition:
            filtered_entities.append(entity)
            filtered_spans.append(span)
            filtered_probabilities.append(probability)

    return filtered_entities, filtered_spans, filtered_probabilities



def get_entities_from_tags(tags: List[str], 
                           offset_mapping: np.ndarray, 
                           probabilities: Optional[np.ndarray]=None,
                           confidence_func=lambda probabilities: probabilities.mean(),
                           ) -> Tuple[List[str], List[List[int]], List[float]]:
    
    length = len(offset_mapping)
    
    if probabilities is None:
        probabilities = np.zeros(length)

    entities, spans, confidences, entity, i = [], [], [], None, None
    for j, tag in enumerate(tags):
        if entity is not None and tag != f"I-{entity}":
            span = [offset_mapping[i][0], offset_mapping[j - 1][1]]
            entities.append(entity)
            spans.append(span)

            confidence = confidence_func(probabilities[i:j])
            confidences.append(confidence)

            entity = None
        if tag.startswith("B-"):
            entity, i = tag[2:], j

    # Because BIO-naming does not ensure the end of the entities (i.e. E-tag), we cannot
    # automatically detect the end of the last entity in the above loop.
    if entity is not None:
        span = [offset_mapping[i][0], offset_mapping[-1][1]]
        entities.append(entity)
        spans.append(span)

        confidence = confidence_func(probabilities[i:])
        confidences.append(confidence)

    return entities, spans, confidences



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