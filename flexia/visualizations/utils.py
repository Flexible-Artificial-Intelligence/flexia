from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, List


from ..import_utils import is_spacy_available, is_seaborn_available, is_matplotlib_available
from ..optimization_utils import get_stepped_lrs


if is_matplotlib_available():
    import matplotlib.pyplot as plt

if is_seaborn_available():
    import seaborn as sns

if is_spacy_available():
    import spacy


def display_text(text:str, 
                 entities:Optional[List[str]]=None, 
                 spans:Optional[List[List[int]]]=None, 
                 colors:Dict[str, str]={}, 
                 title:Optional[str]=None, 
                 *args, 
                 **kwargs) -> None:
    ents = []
    if entities is not None and spans is not None:
        for (start, end), entity in zip(spans, entities):
            ents.append({"start": start, "end": end, "label": entity})

    document = {
        "text" : text,
        "ents" : ents,
        "title": title,
    }

    options = {"colors": colors}
    spacy.displacy.render(docs=document, options=options, *args, **kwargs)


def plot_lr(optimizer:Optimizer, 
            scheduler:Optional[_LRScheduler]=None, 
            steps:int=100, 
            only_last_group:bool=True, 
            key:str="lr"
            ) -> None:
            
    steps, groups_lrs = get_stepped_lrs(optimizer=optimizer, 
                                        scheduler=scheduler, 
                                        steps=steps, 
                                        steps_start=1, 
                                        return_steps_list=True,
                                        only_last_group=only_last_group, 
                                        key=key)
    
    
    figure = plt.figure()
    ax = figure.add_subplot()
    
    if only_last_group:
        sns.lineplot(x=steps, y=groups_lrs, ax=ax)
    else:
        for group_index, group_lrs in enumerate(groups_lrs):
            sns.lineplot(x=steps, y=group_lrs, label=group_index, ax=ax)
        
    figure.legend()    
    figure.tight_layout()
    figure.show()