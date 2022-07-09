import torch
from torch import nn


class MeanPooling(nn.Module):
    def __init__(self, eps=1e-9):
        super(MeanPooling, self).__init__()
        self.eps = eps

    def forward(self, hidden_state:torch.Tensor, attention_mask:torch.Tensor):
        hidden_size = hidden_state.size()
        
        if len(attention_mask.size()) != 3:
            attention_mask = attention_mask.unsqueeze(dim=-1) # torch.Size([batch_size, max_length, 1])
        
        # expanding attention mask to the hidden state's size
        attention_mask_expanded = attention_mask.expand(hidden_size).float() # torch.Size([batch_size, max_length, hidden_size])
        
        # multiply hidden state with attention mask, in order to prevent computing mean with PAD tokens, then get mean. 
        sum_embeddings = torch.sum(hidden_state * attention_mask_expanded, dim=1) # torch.Size([batch_size, hidden_size])
        sum_mask = attention_mask_expanded.sum(dim=1) # torch.Size([batch_size, hidden_size])
        sum_mask = torch.clamp(sum_mask, min=self.eps) # torch.Size([batch_size, hidden_size])
        mean_embeddings = (sum_embeddings / sum_mask) # torch.Size([batch_size, hidden_size])
        return mean_embeddings