from typing import Optional, Union

import torch
from torch.nn import Linear, Embedding
from transformers import Gemma2Config, Gemma2Model, Gemma2ForCausalLM
from transformers.cache_utils import HybridCache

from models.abstract_models import AbsModelConfig, AbsModel


class Gemma2RRecConfig(Gemma2Config, AbsModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Gemma2RRecCasualLM(Gemma2ForCausalLM):
    config_class = Gemma2RRecConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(config.hidden_size,
                              config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[HybridCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            return_causal_output: bool = True,
            return_with_last_hidden_states: bool = False,
            **loss_kwargs,

    ):
        assert return_causal_output or return_with_last_hidden_states, \
            "At least one of return_causal_output or return_with_last_hidden_states must be True."

        if not return_causal_output:
            result = AbsModel.forward(self,
                                            attention_mask=attention_mask,
                                            input_ids=input_ids,
                                            position_ids=position_ids, )
   
            return result
                
            
        if return_with_last_hidden_states:
            output_hidden_states = True
            return_dict = True

        causallm_output = Gemma2ForCausalLM.forward(self,
                                                    input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    position_ids=position_ids,
                                                    past_key_values=past_key_values,
                                                    inputs_embeds=inputs_embeds,
                                                    labels=labels,
                                                    use_cache=use_cache,
                                                    output_attentions=output_attentions,
                                                    output_hidden_states=output_hidden_states,
                                                    return_dict=return_dict,
                                                    cache_position=cache_position,
                                                    logits_to_keep=logits_to_keep,
                                                    **loss_kwargs,
                                                    )
        if not return_with_last_hidden_states:
            return causallm_output
        
        # tuple(torch.FloatTensor)[-1] ->  (B, S, hidden_size)
        last_block_hidden_states = causallm_output.hidden_states[-1]

        # (B, hidden_size)
        last_token_hidden_states = AbsModel.forward(self,
                                                          attention_mask=attention_mask,
                                                          input_ids=input_ids,
                                                          hidden_states=last_block_hidden_states,
                                                          )
        return causallm_output, last_token_hidden_states
