from datetime import datetime
from typing import Optional

import datasets

from prompters.abstract_prompter import AbstractPrompter
from prompters.prompts import obtain_prompts

DESCRIPTION_MAX_LEN = 100


def get_item_info_str(sequence, item_dset, just_title=False):
    if just_title:
        item_info_str = sequence['item_title'] if "item_title" in sequence else sequence['title']
        return item_info_str
    if "title" in sequence:  # this is item_info split
        item_title = sequence['title']
        item_info = sequence
    else:  # this is trn/test/val split
        item_id = sequence["seq_labels"]
        item_title = sequence["item_title"]
        item_info = item_dset[item_id - 1]
        item_title_2 = item_info['title']
        assert item_title == item_title_2, f"item_title: {item_title}, item_title_2: {item_title_2}"

    average_rating = item_info['average_rating']
    num_buyers = item_info['rating_number']
    description = item_info['description']

    description = "" if len(description) == 0 else ' '.join(description[::-1])
    des = description.split()
    len_des = len(des)
    if len_des > DESCRIPTION_MAX_LEN:
        description = ' '.join(des[:DESCRIPTION_MAX_LEN]) + '...'

    item_info_str = (f"Title: {item_title}\n"
                     f"User Rating: {average_rating}\n"
                     f"Number of Buyers: {num_buyers}\n"
                     f"Description: {description}")
    # item_info_str = item_title
    return item_info_str


def format_timedelta(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    minutes += seconds / 60
    if days > 0:
        # turn hours, minutes into float hours
        hours += minutes / 60
        _str = f"{days}d {hours:.1f}h"
    else:
        if hours > 0:
            _str = f"{hours}h {minutes:.1f}min"
        else:
            _str = f"{minutes:.1f}min"
    _str += ' ago'
    return _str


def timestamp2str(example):
    history_timestamp = example['history_timestamp']
    timestamp = example['timestamp']
    history_timestamp = [datetime.fromtimestamp(
        t/1000) for t in history_timestamp]
    timestamp = datetime.fromtimestamp(timestamp/1000)
    delta_times = [timestamp - t for t in history_timestamp]
    human_readable = [format_timedelta(delta) for delta in delta_times]
    return human_readable


def get_items_str(sequence, item_dset, win_size=10, just_title=False):
    strs = []
    dset_win_size = len(sequence['seq_input_ids'])
    _time_delta_strs = timestamp2str(sequence)
    # assert win_size <= dset_win_size, f"win_size: {win_size}, dset_win_size: {dset_win_size}"
    start = 0 if dset_win_size <= win_size else dset_win_size - win_size
    for i in range(start, dset_win_size):
        _title = sequence['history_item_title'][i]
        _rating = sequence['history_rating'][i]
        if just_title:
            strs.append(
                f"{_time_delta_strs[i]}: [{_title}] ({_rating})")
            continue
        item_id = sequence['seq_input_ids'][i]
        item_info = item_dset[item_id - 1]
        description = item_info['description']
        description = "" if len(
            description) == 0 else ' '.join(description[::-1])
        des = description.split()
        len_des = len(des)
        if len_des > DESCRIPTION_MAX_LEN:
            description = ' '.join(des[:DESCRIPTION_MAX_LEN]) + '...'
        if len(description) > 0:
            description = '\n\t' + description
        strs.append(f"{i + 1 - start}.\t{_title} "
                    f"({_rating})" + description)
    history_items = '\n'.join(strs)
    return history_items


class UserGenPrompter(AbstractPrompter):
    user_content_key = "prompt"
    sys_content_key = None

    def __init__(self,
                 tokenizer,
                 category,
                 dset=None,
                 window_size=10,
                 input_ids_max_length=2048,
                 emb_token='',
                 emb_end_token='',
                 ):

        super().__init__(tokenizer)
        self.dset = dset
        self.window_size = window_size
        self.input_ids_max_length = input_ids_max_length
        
        self.emb_token = emb_token or self.tokenizer.generation_end
        self.emb_end_token = emb_end_token

        prompts = obtain_prompts(category)
        self.user_analyze_prompt = prompts['user_prompt'].format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token)

    def to_chat_example(self, sequence):
        history_items = get_items_str(
            sequence, self.dset['item_info'], self.window_size, just_title=True)
        sequence['prompt'] = self.user_analyze_prompt + '\n' + history_items
        return sequence

    def convert_dataset(self,
                        split: Optional[str] = None,
                        dset: datasets.Dataset = None,
                        return_messages: bool = False,
                        ):

        new_dataset = self.dset[split] if split is not None else dset

        # check window size
        seqs = new_dataset['seq_input_ids']
        seq_max_len = max([len(seq) for seq in seqs])
        assert self.window_size <= seq_max_len, f"Invalid window size: {self.window_size}, seq_max_len: {seq_max_len}"

        new_dataset = new_dataset.map(self.to_chat_example,
                                      desc='Converting to chat examples',
                                      batched=False,
                                      keep_in_memory=True
                                      )

        if return_messages:
            def _to_messages(example):
                example['messages'] = self.formatting_func(
                    example, return_messages=True)
                return example

            return new_dataset.map(_to_messages,
                                   desc='Converting to chat examples',
                                   batched=False,)

        new_dataset = new_dataset.map(self.totensor,
                                      desc='Applying chat template',
                                      batched=True,
                                      fn_kwargs={
                                          'max_length': self.input_ids_max_length,
                                      }
                                      )
        if 'user_gen_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['user_gen_input_ids',
                                                      'user_gen_attention_mask'])

        new_dataset = new_dataset.rename_column(
            'input_ids', 'user_gen_input_ids')
        new_dataset = new_dataset.rename_column(
            'attention_mask', 'user_gen_attention_mask')

        return new_dataset


class ItemPrompter(AbstractPrompter):
    user_content_key = "prompt"
    assistant_content_key = "assistant"
    
    def __init__(self,
                 tokenizer,
                 category,
                 dset=None,
                 input_ids_max_length=512,
                 emb_token='',
                 emb_end_token='',
                 ):
        super().__init__(tokenizer)
        self.dset = dset
        self.input_ids_max_length = input_ids_max_length
        self.emb_token = emb_token or self.tokenizer.generation_end
        self.emb_end_token = emb_end_token

        self.prompts = obtain_prompts(category)

        self.prompts['item_prompt'] = self.prompts['item_prompt'].format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token,
        )

    def convert_dataset(self,
                        split: Optional[str] = None,
                        dset: datasets.Dataset = None,
                        ):
        new_dataset = self.dset[split] if split is not None else dset

        new_dataset = new_dataset.map(self.to_chat_example_item,
                                      desc=f'Converting to chat examples for item profile',
                                      batched=False,
                                      keep_in_memory=True,
                                      )
        new_dataset = new_dataset.map(self.totensor,
                                      desc=f'Applying chat template for item profile',
                                      batched=True,
                                      fn_kwargs={
                                          'max_length': self.input_ids_max_length,
                                          'continue_final_message': True,
                                      }
                                      )
        if 'item_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['item_input_ids',
                                                      'item_attention_mask'])
        new_dataset = new_dataset.rename_column('input_ids', 'item_input_ids')
        new_dataset = new_dataset.rename_column(
            'attention_mask', 'item_attention_mask')

        return new_dataset

    def to_chat_example_item(self, sequence):
        item_info_str = get_item_info_str(
            sequence, self.dset['item_info'], just_title=False)
        sequence['prompt'] = self.prompts['item_prompt'] + \
            '\n\n' + item_info_str
        sequence['assistant'] = self.emb_token
        return sequence


class UserPrompter(AbstractPrompter):
    assistant_content_key = 'completion_0'
    user_content_key = "prompt"
    sys_content_key = None

    def __init__(self,
                 tokenizer,
                 category,
                 dset=None,
                 window_size=10,
                 input_ids_max_length=1024,
                 emb_token='',
                 emb_end_token='',
                 ):
        super().__init__(tokenizer)
        self.dset = dset
        self.window_size = window_size
        self.input_ids_max_length = input_ids_max_length
        self.emb_token = emb_token or self.tokenizer.generation_end
        self.emb_end_token = emb_end_token

        self.prompts = obtain_prompts(category)
        self.prompts['user_prompt'] = self.prompts['user_prompt'].format(
            emb_token=self.emb_token,
            emb_end_token=self.emb_end_token,
        )

    def convert_dataset(
        self,
        split: Optional[str] = None,
        dset: datasets.Dataset = None,
    ):
        new_dataset = self.dset[split] if split is not None else dset

        assert split is None or split != 'item_info'
        # new_dataset = new_dataset.filter(lambda x: len(x['profile']) > 0)
        # assert len(new_dataset) > 0, f"Empty dataset for split {split}"

        # convert user profile to chat example
        new_dataset = new_dataset.map(self.to_chat_example,
                                      desc='Converting to chat examples for user profile',
                                      batched=False,
                                      keep_in_memory=True
                                      )
        new_dataset = new_dataset.map(self.totensor,
                                      desc='Applying chat template for user profile',
                                      batched=True,
                                      fn_kwargs={
                                          'max_length': self.input_ids_max_length,
                                          'continue_final_message': True
                                      }
                                      )
        if 'user_input_ids' in new_dataset.column_names:
            new_dataset = new_dataset.remove_columns(['user_input_ids',
                                                      'user_attention_mask'])
        new_dataset = new_dataset.rename_column(
            'input_ids', 'user_input_ids')
        new_dataset = new_dataset.rename_column(
            'attention_mask', 'user_attention_mask')

        return new_dataset

    def to_chat_example(self, sequence):
        history_items = get_items_str(sequence, self.dset["item_info"], self.window_size, just_title=True)
        prompt = self.prompts["user_prompt"]
        sequence["prompt"] = prompt + "\n" + history_items

        if isinstance(sequence['profile'], str):
            sequence['profile'] = [sequence['profile']]
        if isinstance(sequence['profile'], list):
            for i, profile in enumerate(sequence['profile']):
                if self.emb_token != self.tokenizer.generation_end:
                    if self.emb_token not in profile:
                        profile = profile + self.emb_token
                sequence[f"completion_{i}"] = profile
        else:
            raise ValueError(f"reasoning is not a string or list: {sequence['profile']}")
        return sequence

    def totensor_multiple(self, element):
        outputs = self.tokenizer(
            [self.formatting_func(element,
                                  completions_key=f"completion_{i}",
                                  continue_final_message=True
                                  ) for i in range(len(element['profile']))],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=self.input_ids_max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        result = {"multi_user_input_ids": outputs["input_ids"],
                  "multi_user_attention_mask": outputs["attention_mask"],
                  }
        
        result["multi_user_completion_range"] = self.find_completion_start_end(outputs["input_ids"])

        return result

    def find_final_start_end(self, input_ids: list[list[int]], start_patterns: list[str]):
        # 为了 1. data_collator 里面 mask 无关位置的labels方便一点
        #     2. 可以方便data_collator里计算一下前多少tokens不用送LMHead

        in_lens = [len(input_id) for input_id in input_ids]

        ranges = []
        batch_size = len(in_lens)
        for i in range(batch_size):
            # find the start pattern from the end
            current_input_ids = input_ids[i]
            current_in_len = in_lens[i]
            current_start_pattern = start_patterns[i]
            if current_start_pattern is None:
                ranges.append(None)
                continue
            current_start_pattern = self.tokenizer.encode(current_start_pattern, add_special_tokens=False)
            _start_len = len(current_start_pattern)
            start_index = None
            for idx in range(current_in_len, -1, -1):
                if current_input_ids[idx : idx + _start_len] == current_start_pattern:
                    start_index = idx
                    break
            if start_index is None:
                raise ValueError(
                    f"start pattern [{self.tokenizer.decode(current_start_pattern)}] not found in input_ids:\n {self.tokenizer.decode(input_ids[i])}"
                )
            result = (start_index - 1, start_index + _start_len)

            ranges.append(result)

        return ranges
