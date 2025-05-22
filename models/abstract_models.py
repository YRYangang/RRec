from typing import Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast


class AbsModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AbsModel(PreTrainedModel):
    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                input_ids: torch.LongTensor = None,
                hidden_states: Optional[torch.FloatTensor] = None,
                **model_kwargs,
                ):
        if hidden_states is None:
            transformer_outputs: BaseModelOutputWithPast = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                **model_kwargs,
            )
            # (B, S, hidden_size) from the last(top) transformer block
            hidden_states = transformer_outputs[0]
        else:
            assert hidden_states.shape[0] == input_ids.shape[0]
            assert hidden_states.shape[1] == input_ids.shape[1]
            assert hidden_states.shape[2] == self.config.hidden_size
            
        batch_size = input_ids.shape[0]

        sequence_lengths = torch.eq(
            input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(hidden_states.device)

        last_hidden_states = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), sequence_lengths]
        return last_hidden_states
