from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling
from trl.trainer.utils import pad


@dataclass
class RRecDataCollator(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {}
        feature_keys = examples[0].keys()
        for key in feature_keys:
            if "input_ids" not in key:
                continue
            if "multi" in key:
                continue
            attn_key = key.replace("input_ids", "attention_mask")
            if attn_key not in feature_keys:
                continue
            self.tokenizer.padding_side = "left" if 'gen' in key else "right"
            examples_to_encode = [
                {
                    "input_ids": e[key],
                    "attention_mask": e[attn_key]
                } for e in examples]
            inputs = super().torch_call(examples_to_encode)
            batch[key] = inputs["input_ids"]
            batch[attn_key] = inputs["attention_mask"]
            
            if 'gen' in key:
                batch[key+'_no_pad'] = [e[key] for e in examples]
            
            completion_range_key = key.replace("input_ids", "completion_range")
            if completion_range_key in feature_keys:
                labels = torch.full_like(inputs["input_ids"], -100)
                for i, e in enumerate(examples):
                    assert completion_range_key in e
                    if e[completion_range_key] is None:
                        continue
                    start, end = e[completion_range_key]
                    labels[i, start:end] = inputs["input_ids"][i, start:end]
                
                labels_key = key.replace("input_ids", "labels")
                batch[labels_key] = labels

        for key in feature_keys:
            if "input_ids" not in key and "id" in key:
                batch[key]: List[int | str] = [e[key] for e in examples]

        if "seq_labels" in examples[0]:
            batch["seq_labels"] = torch.tensor(
                [e["seq_labels"] for e in examples], dtype=torch.long)

        if "seq_input_ids" in examples[0]:
            seq_input_ids = [torch.tensor(
                e["seq_input_ids"], dtype=torch.long) for e in examples]
            batch["seq_input_ids"] = pad(
                seq_input_ids,
                0,
                'right',
            )
        if "multi_user_input_ids" in examples[0]:
            batch |= self.handle_nested_inputs(examples, prefix='multi_user')

        if "multi_item_input_ids" in examples[0]:
            batch |= self.handle_nested_inputs(examples, prefix='multi_item')
        
        assert len(batch), "No valid input keys found in the examples"
        return batch

    def handle_nested_inputs(self, examples, prefix='multi_user'):
        """
        handle multi_user_input_ids
        """
        self.tokenizer.padding_side = "right"
        examples_to_encode = []
        completion_ranges = []
        for e in examples:
            for i in range(len(e[f"{prefix}_input_ids"])):
                examples_to_encode.append(
                    {
                        "input_ids": e[f"{prefix}_input_ids"][i],
                        "attention_mask": e[f"{prefix}_attention_mask"][i]
                    }
                )
                if f"{prefix}_completion_range" in e:
                    completion_ranges.append(
                        e[f"{prefix}_completion_range"][i])

        inputs = super().torch_call(examples_to_encode)
        labels = torch.full_like(inputs["input_ids"], -100)
        
        if len(completion_ranges):
            for i, _range in enumerate(completion_ranges):
                if _range is not None:
                    start, end = _range
                    labels[i, start:end] = inputs["input_ids"][i, start:end]

        result = {
            f"{prefix}_input_ids": inputs["input_ids"],
            f"{prefix}_attention_mask": inputs["attention_mask"],
            f"{prefix}_labels": labels,
        }

        return result

