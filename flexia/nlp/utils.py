from typing import List, Any, Tuple, Callable
import numpy as np

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




def get_words_offset_mapping(text: str, 
                             split_function: Callable[[str], List[str]] = lambda text: text.split()
                             ) -> List[Tuple[int, int]]:
    
    """
    Gets offset mapping for words in the text.
    """
    
    words = split_function(text)
    
    words_offset_mapping = []
    word_start_index, word_end_index = 0, 0
    for word in words:
        word_length = len(word)
        word_end_index = word_start_index + word_length
        word_span = (word_start_index, word_end_index)
        words_offset_mapping.append(word_span)

        word_start_index = word_end_index + 1
        
    return words_offset_mapping


def convert_tokens_to_chars_predictions(text: str, 
                                        tokens_predictions: Any, 
                                        offset_mapping: List[Tuple[int, int]],
                                        ) -> np.ndarray:
    """
    Converts tokens predictions to chars predictions.     
    """
    
    text_length = len(text)
    chars_predictions = np.zeros(shape=text_length, dtype=np.float32)
    
    for token_index, (start, end) in enumerate(offset_mapping):
        token_prediction = tokens_predictions[token_index]
        chars_predictions[start:end] = token_prediction
    
    return chars_predictions



def convert_chars_to_tokens_predictions(chars_predictions: Any, 
                                        offset_mapping: List[Tuple[int, int]],
                                        aggregation_function: Callable[[Any], Any] = lambda x: np.mean(x),
                                        ) -> np.ndarray:
    """
    Converts chars predictions to tokens predictions.
    """
    
    num_tokens = len(offset_mapping)
    tokens_predictions = np.zeros(shape=num_tokens, dtype=np.float32)
    
    for token_index, (start, end) in enumerate(offset_mapping):
        token_predictions = aggregation_function(chars_predictions[start:end])
        tokens_predictions[token_index] = token_predictions
        
    return tokens_predictions


def convert_words_to_chars_predictions(text: str, 
                                       words_predictions: Any, 
                                       words_offset_mapping: List[Tuple[int, int]],
                                       ) -> np.ndarray:
    """
    Converts words predictions to chars predictions.
    """
    
    text_length = len(text)
    chars_predictions = np.zeros(shape=text_length, dtype=np.float32)
    
    for word_index, (start, end) in enumerate(words_offset_mapping):
        word_prediction = words_predictions[word_index]
        chars_predictions[start:end] = word_prediction
    
    return chars_predictions


def convert_chars_to_words_predictions(chars_predictions:Any, 
                                       words_offset_mapping: List[Tuple[int, int]],
                                       aggregation_function: Callable[[Any], Any] = lambda x: np.mean(x)
                                       ) -> np.ndarray:
    """
    Converts chars predictions to words predictions.
    """
    
    num_words = len(words_offset_mapping)
    words_predictions = np.zeros(shape=num_words, dtype=np.float32)
    
    for word_index, (start, end) in enumerate(words_offset_mapping):
        word_predictions = aggregation_function(chars_predictions[start:end])
        words_predictions[word_index] = word_predictions
        
    return words_predictions


def convert_tokens_to_words_predictions(text: str, 
                                        tokens_predictions: Any, 
                                        offset_mapping: List[Tuple[int, int]], 
                                        words_offset_mapping: List[Tuple[int, int]], 
                                        aggregation_function: Callable[[Any], Any] = lambda x: np.mean(x),
                                        ):
    """
    Converts tokens predictions to words predictions.
    """
    
    # convert tokens to chars predictions
    chars_predictions = convert_tokens_to_chars_predictions(text=text, 
                                                            tokens_predictions=tokens_predictions, 
                                                            offset_mapping=offset_mapping)
    
    
    # convert chars predictions to words predictions
    words_predictions = convert_chars_to_words_predictions(chars_predictions=chars_predictions, 
                                                           words_offset_mapping=words_offset_mapping, 
                                                           aggregation_function=aggregation_function)
    
    return words_predictions

def convert_words_to_tokens_predictions(text: str, 
                                        words_predictions: Any, 
                                        words_offset_mapping: List[Tuple[int, int]], 
                                        offset_mapping: List[Tuple[int, int]], 
                                        aggregation_function: Callable[[Any], Any] =lambda x: np.mean(x)
                                        ):
    """
    Converts words predictions to tokens predictions.
    """
    
    # convert words predictions to chars predictions
    chars_predictions = convert_words_to_chars_predictions(text=text, 
                                                           words_predictions=words_predictions, 
                                                           words_offset_mapping=words_offset_mapping)
    
    
    # convert chars predictions to tokens predictions
    tokens_predictions = convert_chars_to_tokens_predictions(chars_predictions, 
                                                             offset_mapping=offset_mapping, 
                                                             aggregation_function=aggregation_function)
    
    return tokens_predictions