from typing import List


class AbstractPrompter:
    user_content_key = None
    assistant_content_key = None
    sys_content_key = None
    add_completion_range = False

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer
        # check if tokenizer has attribute generation_prompt
        if hasattr(self.tokenizer, "generation_prompt"):
            self.generations = [
                self.tokenizer.encode(self.tokenizer.generation_prompt, add_special_tokens=False),
                self.tokenizer.encode(self.tokenizer.generation_end, add_special_tokens=False),
            ]
        else:
            assert (
                not self.add_completion_range
            ), "If add_completion_range is True, tokenizer must have generation_prompt"

    def formatting_func(
        self,
        examples,
        prompt_key=None,
        completions_key=None,
        sys_prompt_key=None,
        return_messages=False,
        **kwargs,
    ):

        prompt_key = prompt_key or self.user_content_key
        completions_key = completions_key or self.assistant_content_key
        sys_prompt_key = sys_prompt_key or self.sys_content_key

        add_generation_prompt = completions_key is None

        if isinstance(examples[prompt_key], list):
            conversations = []
            for i in range(len(examples[prompt_key])):
                conversation = []
                if sys_prompt_key is not None:
                    conversation += [{"role": "system", "content": examples[sys_prompt_key][i]}]

                conversation += [{"role": "user", "content": examples[prompt_key][i]}]

                if completions_key is not None:
                    conversation += [{"role": "assistant", "content": examples[completions_key][i]}]
                if not return_messages:
                    conversation = self.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
                    )

                conversations.append(conversation)

            return conversations
        else:
            conversation = []
            if sys_prompt_key is not None:
                conversation += [{"role": "system", "content": examples[sys_prompt_key]}]

            conversation += [{"role": "user", "content": examples[prompt_key]}]

            if completions_key is not None:
                conversation += [{"role": "assistant", "content": examples[completions_key]}]

            if not return_messages:
                conversation = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
                )

            return conversation

    def totensor(self, element, max_length=4096, **kwargs):
        outputs = self.tokenizer(
            self.formatting_func(element, **kwargs),
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        result = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
        if self.assistant_content_key is not None and self.add_completion_range:
            result |= {"completion_range": self.find_completion_start_end(outputs["input_ids"])}
        return result

    def find_completion_start_end(self, input_ids: List[List[int]]):

        start_pattern, end_pattern = self.generations
        start_len = len(start_pattern)
        end_len = len(end_pattern)
        in_lens = [len(input_id) for input_id in input_ids]

        ranges = []
        batch_size = len(in_lens)
        for i in range(batch_size):
            # find the start pattern from the end
            current_input_ids = input_ids[i]
            current_in_len = in_lens[i]
            start_index = None
            for idx in range(current_in_len - start_len - end_len, -1, -1):
                if current_input_ids[idx : idx + start_len] == start_pattern:
                    start_index = idx
                    break
            if start_index is None:
                raise ValueError(
                    f"start pattern [{self.tokenizer.decode(start_pattern)}] "
                    f"not found in input_ids:\n {self.tokenizer.decode(input_ids[i])}"
                )
            if end_pattern not in current_input_ids[start_index + start_len :]:
                result = (start_index + start_len, current_in_len)
            else:
                for j in range(start_index + start_len, current_in_len - end_len):
                    if current_input_ids[j : j + end_len] == end_pattern:
                        result = (start_index + start_len, j + 1)
                        break
            ranges.append(result)

        return ranges
