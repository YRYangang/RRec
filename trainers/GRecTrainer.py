import warnings
import os
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Dict, Callable, Optional, Union, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.amp as amp

from transformers.utils import logging
from transformers.trainer_utils import seed_worker
from transformers.integrations import deepspeed_init
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback, GenerationConfig
from transformers import Trainer, TrainingArguments
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax
from trl.trainer import disable_dropout_in_model
from accelerate.utils.other import is_compiled_module
from accelerate.utils import is_peft_model
import datasets

from vllm import LLM, SamplingParams, TokensPrompt
import rich

from unittest.mock import patch
from tqdm import tqdm


from prompters.rrec_prompter import UserGenPrompter, ItemPrompter, UserPrompter
from trainers.utils import Similarity

logger = logging.get_logger(__name__)


@dataclass
class GenRecTrainingArguments(TrainingArguments):
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["seq_labels"],
        metadata={
            "help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    emb_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    emb_end_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    user_input_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The max length of user input."},
    )
    item_input_max_length: Optional[int] = field(
        default=768,
        metadata={"help": "The max length of item input."},
    )
    generation_config: GenerationConfig = field(
        default=None,
        metadata={"help": "The generation config."},
    )
    test_generation_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The batch size for generation during evaluation."},
    )
    train_generation_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The batch size for generation during training."},
    )
    print_out_examples: Optional[int] = field(
        default=10,
        metadata={"help": "Print out examples every n steps."},
    )
    mini_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "The mini batch size Rec Training."},
    )
    item_emb_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for embedding item profiles for evaluation."},
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Do sampling when generating user and item reasoning."},
    )
    dataset_category: Optional[str] = field(
        default='',
        metadata={"help": "The category of the dataset."},
    )
    dataset_window_size: Optional[int] = field(
        default=4,
        metadata={"help": "The window size for user and item input."},
    )
    disable_dropout: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable dropout during training."},
    )
    similarity_type: str = field(
        default="dot",
        metadata={
            "help": "The type of similarity function to use. Options: dot, cosine, L2."},
    )
    similarity_temperature: float = field(
        default=0.02,
        metadata={"help": "The temperature for the similarity function."},
    )
    similarity_normalization: bool = field(
        default=True,
        metadata={
            "help": "Whether to normalize the similarity scores before computing similarity."},
    )
    gather_negs_across_processes: bool = field(
        default=True,
        metadata={"help": "Whether to gather negative samples across processes."},
    )

    use_vllm: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use vllm for generation."},
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )


class GenRecTrainer(Trainer):

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],

            args: GenRecTrainingArguments,
            data_collator: Optional[DataCollator],
            full_dataset: Optional["datasets.Dataset"],

            processing_class: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_loss_func: Optional[Callable] = None,
            compute_metrics: Optional[Callable] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers=(None, None),
            preprocess_logits_for_metrics: Optional[Callable[[
                torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):

        self.args = args

        self.args.mini_batch_size = min(
            self.args.mini_batch_size, self.args.per_device_train_batch_size)

        window_size = args.dataset_window_size
        emb_token = args.emb_token
        emb_end_token = args.emb_end_token
        self.get_user_profile_prompter = UserGenPrompter(dset=full_dataset,
                                                         tokenizer=processing_class,
                                                         category=args.dataset_category,
                                                         window_size=window_size,
                                                         emb_token=emb_token,
                                                         emb_end_token=emb_end_token
                                                         )

        user_prompter = UserPrompter(dset=full_dataset,
                                     tokenizer=processing_class,
                                     category=args.dataset_category,
                                     window_size=window_size,
                                     input_ids_max_length=args.user_input_max_length,
                                     emb_token=emb_token,
                                     emb_end_token=emb_end_token
                                     )

        item_prompter = ItemPrompter(dset=full_dataset,
                                     tokenizer=processing_class,
                                     category=args.dataset_category,
                                     input_ids_max_length=args.item_input_max_length,
                                     emb_token=emb_token,
                                     emb_end_token=emb_end_token
                                     )

        self.item_prompter = item_prompter
        self.user_prompter = user_prompter
        print("Prompters initialized, start converting datasets...", end='')

        for split in ['train', 'valid', 'test']:
            full_dataset[split] = full_dataset[split].rename_columns({'history_item_id': 'seq_input_ids',
                                                                      'item_id': 'seq_labels'})
            full_dataset[split] = self.get_user_profile_prompter.convert_dataset(
                dset=full_dataset[split])
            full_dataset[split] = item_prompter.convert_dataset(
                dset=full_dataset[split])
            # full_dataset[split] = self.get_item_profile_prompter.convert_dataset(dset=full_dataset[split])

        full_dataset['item_info'] = item_prompter.convert_dataset(
            dset=full_dataset['item_info'])


        # add a idx column to the train dataset
        full_dataset['train'] = full_dataset['train'].add_column("train_data_id", 
                                                                 range(len(full_dataset['train'])))
        train_dataset = full_dataset['train']

        self.item_dataset = full_dataset['item_info']
        if getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            assert hasattr(model, "enable_input_require_grads")
            model.enable_input_require_grads()

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.previous_log = {}

        self.is_peft_model = is_peft_model(model)

        if args.disable_dropout:
            disable_dropout_in_model(model)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset={
                'test': full_dataset['test'], 'valid': full_dataset['valid']},
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.item_hs = None
        self.similarity = Similarity(self.args)

        if args.use_vllm:
            if self.accelerator.is_main_process:

                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"
                    self.accelerator.print(
                        f"Using vLLM on device {vllm_device}")
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                    # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        tokenizer=self.args.output_dir,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # enable_prefix_caching=True,
                        device=vllm_device,
                        # When release by vLLM, we would be able to distribute the model on multiple GPUs
                        # See https://github.com/vllm-project/vllm/pull/12071
                        # tensor_parallel_size=torch.cuda.device_count(),
                        # distributed_executor_backend="external_launcher",
                    )
                    self.vllm_sampling_params = SamplingParams(
                        temperature=self.args.generation_config.temperature,
                        max_tokens=self.args.generation_config.max_new_tokens,
                        top_k=self.args.generation_config.top_k,
                        top_p=self.args.generation_config.top_p,
                        repetition_penalty=self.args.generation_config.repetition_penalty,
                        n=self.args.generation_config.num_return_sequences,
                        seed=self.args.seed,

                        include_stop_str_in_output=True,
                        stop=[emb_token] if emb_token else None,
                    )

                    self.vllm_no_sampling_params = SamplingParams(
                        temperature=0,
                        max_tokens=self.args.generation_config.max_new_tokens,
                        repetition_penalty=self.args.generation_config.repetition_penalty,

                        include_stop_str_in_output=True,
                        stop=[emb_token] if emb_token else None,
                    )
                    rich.print("[red]VLLM sampling params[/red]")
                    rich.print(self.vllm_sampling_params)

            self._last_loaded_step = 0  # tag to avoid useless loading during grad acc
            self.accelerator.wait_for_everyone()
            print('*'*20 + ' VLLM initialized' + '*'*20)
            self._move_model_to_vllm()
            print('*'*20 + ' VLLM moved to vllm' + '*'*20)

    def _generate_item_embeddings(self, model,
                                  input_prefix: str = "item",
                                  ):
        with torch.no_grad():
            dataloader = self._get_item_embed_dataloader()
            all_item_hs = []
            pbar = tqdm(
                dataloader, desc=f"[cuda:{self.accelerator.process_index}] Generating item embeddings")
            for batch in pbar:
                item_id = batch["item_id"]
                batch = self._prepare_inputs(batch)
                item_h = model(attention_mask=batch[f"{input_prefix}_attention_mask"],
                               input_ids=batch[f"{input_prefix}_input_ids"],
                               return_with_last_hidden_states=True,
                               return_causal_output=False,
                               )

                all_item_hs.append({
                    "item_h": item_h,
                    "item_id": item_id,
                })
            self.accelerator.wait_for_everyone()
            item_hs = {}
            # clean up cache
            torch.cuda.empty_cache()
            all_item_hs = self.accelerator.gather_for_metrics(all_item_hs)
            for item_batch_dict in all_item_hs:
                for batch_idx in range(len(item_batch_dict["item_id"])):
                    item_id = item_batch_dict["item_id"][batch_idx]
                    item_h = item_batch_dict["item_h"][batch_idx]
                    item_hs[item_id] = item_h.to(self.accelerator.device)
            assert len(item_hs) == len(
                self.item_dataset), f"{len(self.item_hs)} != {len(self.item_dataset)}"
            item_ids = self.item_dataset['item_id']
            item_ids = sorted(item_ids)
            item_hs = [item_hs[item_id].to(
                self.accelerator.device) for item_id in item_ids]
            self.item_hs = torch.stack(item_hs)  # (num_items, hidden_size)

        assert self.item_hs.shape[0] == len(
            self.item_dataset), f"{self.item_hs.shape[0]} != {len(self.item_dataset)}"

    def get_model_for_eval(self, ):
        assert not self.args.jit_mode_eval, "JIT mode is not supported for evaluation."
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        model = self.accelerator.unwrap_model(self.model)

        # while ``train`` is running, cast it to the right dtype first and then put on device
        args = self.args
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        if args.past_index >= 0:
            self._past = None

        return model

    def _gather_and_cat(self, tensor):
        """
        Gathers tensors and concatenates them on every process.
        This is needed for the multi-processing case.

        Args:
            tensor (torch.Tensor): Tensor to gather and concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        num_processes = self.accelerator.num_processes
        tmp_tensor = tensor.detach()
        gathered_tensor = self.accelerator.gather_for_metrics(
            [tmp_tensor], use_gather_object=True)
        current_process = self.accelerator.process_index
        indices_to_take = [i for i in range(
            num_processes) if i != current_process]
        gathered_tensor = [gathered_tensor[i].to(
            tensor.device) for i in indices_to_take]
        gathered_tensor = torch.cat([tensor, *gathered_tensor], dim=0)

        assert gathered_tensor.shape[0] == num_processes * \
            tensor.shape[0], f"{gathered_tensor.shape[0]} != {num_processes * tensor.shape[0]}"
        return gathered_tensor

    def _move_model_to_vllm(self):
        """
        Move the model to vllm
        """
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=True
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items(
                ) if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
                state_dict = {k: v for k, v in state_dict.items()
                              if "extra_head" not in k and "item_embedding" not in k}

            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    def _get_item_embed_dataloader(self):
        dataloader_params = {
            "batch_size": self.args.item_emb_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": False,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "worker_init_fn": seed_worker,
        }

        return self.accelerator.prepare(DataLoader(self.item_dataset, **dataloader_params))

    def _batch_generate(self,
                        model,
                        batch: Dict[str, torch.LongTensor],
                        input_ids_key="input_ids",
                        attn_mask_key="attention_mask",
                        do_sample=True,
                        num_return_sequences=None,
                        **kwargs
                        ) -> List[List[str]]:
        """
        Generate sequences for a batch of inputs.

        Args:
            model: The model to use for generation.
            batch: The batch of inputs.
            input_ids_key: The key in the batch dict for the input ids.
            attn_mask_key: The key in the batch dict for the attention mask.
            do_sample: Whether to sample or not, if True, num_return_sequences == 1.
            num_return_sequences: The number of sequences to return, if sampling.

        Returns:
            A list of lists of generated sequences,
            the outer list length is batch size,
            the inner list length is num_return_sequences.
        """

        ctx = amp.autocast(
            "cuda") if self.args.bf16 else nullcontext()
        with ctx:
            if do_sample:
                assert num_return_sequences
                output = model.generate(
                    input_ids=batch[input_ids_key],
                    attention_mask=batch[attn_mask_key],
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    generation_config=self.args.generation_config,
                    num_return_sequences=num_return_sequences,
                    **kwargs,
                )
            else:
                output = model.generate(
                    input_ids=batch[input_ids_key],
                    attention_mask=batch[attn_mask_key],
                    do_sample=False,
                    pad_token_id=self.processing_class.pad_token_id,
                    generation_config=self.args.generation_config,
                    num_return_sequences=1,
                    **kwargs,
                )

        # output = output[:, batch[input_ids_key].shape[1]:]
        output_decoded = self.processing_class.batch_decode(output[:, batch[input_ids_key].shape[1]:],
                                                            skip_special_tokens=True)
        n_samples = num_return_sequences if do_sample else 1
        output_list = []
        batch_size = batch[input_ids_key].shape[0]
        for i in range(batch_size):
            output_list.append(
                output_decoded[i * n_samples:(i + 1) * n_samples])
        assert len(output_list) == batch_size
        return output_list

    def evaluate(self,
                 eval_dataset: Optional[Union[Dataset,
                                              Dict[str, Dataset]]] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval",
                 ) -> Dict[str, float]:
        """
        During training, when this function is called, the eval_dataset is always None,
        But after override (see source code of Trainer.evaluate()), 
        the eval_dataset will be `self.eval_dataset`, which is a dict, 
        i.e., {'test':a dataset, 'val':a dataset}, and this method will be called twice,
        once for `test` and once for `val`.

        What we do here is to generate the reasoning for both test and val datasets on the initial call.
        And skip this process in the after-override calls.

        So currently, do not manually call this function with eval_dataset setting to something else.
        """
        if eval_dataset is None:
            eval_model = self.get_model_for_eval()
            self._generate_item_embeddings(eval_model)
            os.makedirs(os.path.join(
                self.args.output_dir, 'datasets'), exist_ok=True)

            with torch.no_grad():
                for _set_name in self.eval_dataset.keys():
                    dset_cache_dir = os.path.join(
                        self.args.output_dir, 'datasets', f'reasoning_{_set_name}_{self.state.global_step}')

                    self.eval_dataset[_set_name] = self._update_eval_set(
                        eval_model, self.eval_dataset[_set_name], prefix="user")

                    self.eval_dataset[_set_name].save_to_disk(dset_cache_dir)
                    self.accelerator.print(
                        f"{_set_name} dataset saved to {dset_cache_dir}")

                    self.eval_dataset[_set_name] = self.user_prompter.convert_dataset(
                        dset=self.eval_dataset[_set_name])

                    # calculate reasoning avg length
                    _len = [len(x['user_input_ids']) for x in self.eval_dataset[_set_name]]
                    self.store_metrics({'output_length': sum(_len) / len(_len)},
                                    metric_key_prefix=f"{metric_key_prefix}_{_set_name}")

        eval_output = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        return eval_output

    def _update_eval_set(self, model, eval_dataset, prefix='user'):

        if 'profile' in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns('profile')
        id_key_name = 'interaction_id'

        if self.args.use_vllm:
            if self.accelerator.is_main_process:
                prompts = eval_dataset["user_gen_input_ids"]
                tokens_prompts = [TokensPrompt(
                    prompt_token_ids=prompt) for prompt in prompts]
                outputs = self.llm.generate(
                    tokens_prompts, self.vllm_no_sampling_params)
                outputs = [output.outputs[0].text for output in outputs]
                all_result = {_id: output for _id, output in zip(
                    eval_dataset[id_key_name], outputs)}
            else:
                all_result = {}
            all_result = self.accelerator.gather_for_metrics(
                [all_result], use_gather_object=True)[0]

        else:
            batch_size = self.args.test_generation_batch_size

            data_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(data_loader)
            item_results = {}

            pbar = enumerate(data_loader)
            if self.accelerator.is_main_process:
                pbar = tqdm(pbar, total=len(data_loader),
                            desc=f"Generating {prefix} reasonings")

            for batch_i, batch in pbar:
                result = self._batch_generate(model, batch,
                                              input_ids_key=f"{prefix}_gen_input_ids",
                                              attn_mask_key=f"{prefix}_gen_attention_mask",
                                              do_sample=False,
                                              )
                for i, _id in enumerate(batch[id_key_name]):
                    item_results[_id] = result[i][0]

            all_result_list = self.accelerator.gather_for_metrics(
                [item_results], use_gather_object=True)
            all_result = {}
            for result in all_result_list:
                all_result.update(result)

        assert len(all_result) == len(eval_dataset)

        def map_into_dset(example):
            _id = example[id_key_name]
            example['profile'] = all_result[_id]
            return example

        eval_dataset = eval_dataset.map(map_into_dset)
        return eval_dataset

    def _generate_in_train(self, model, batch,):

        train_data_ids = batch["train_data_id"]
        full_batch_size = batch['seq_labels'].shape[0]
        num_processes = self.accelerator.num_processes

        generation_samples = self.args.generation_config.num_return_sequences

        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # get original conversations

            prompts = batch['user_gen_input_ids_no_pad']
            prompts = self.accelerator.gather_for_metrics(
                [prompts], use_gather_object=True)
            user_results = []
            if self.accelerator.is_main_process:
                # flatten the prompts
                prompts = [
                    prompt for prompts_ in prompts for prompt in prompts_]
                tokens_prompts = [TokensPrompt(
                    prompt_token_ids=prompt) for prompt in prompts]
                if generation_samples == 1:
                    outputs = self.llm.generate(
                        tokens_prompts, self.vllm_no_sampling_params, use_tqdm=False)
                else:
                    outputs = self.llm.generate(
                        tokens_prompts, self.vllm_sampling_params, use_tqdm=False)
                user_results = [[output.outputs[i].text for i in range(
                    self.args.generation_config.num_return_sequences)] for output in outputs]

                # distribute back to each device
            user_results = self.accelerator.gather_for_metrics(
                [user_results], use_gather_object=True)[0]
            assert len(user_results) == full_batch_size * num_processes
            # calculate average lens
            _lens = [len(self.processing_class.encode(
                result, add_special_tokens=False)) for results in user_results for result in results]
            self.store_metrics({'output_len': sum(_lens) / len(_lens)})

            current_process = self.accelerator.process_index
            user_results = [
                user_results[full_batch_size * current_process + i] for i in range(full_batch_size)]
            assert len(user_results) == full_batch_size

        else:
            def _iterator(mini_batch_size, prefix):
                for i in range(0, full_batch_size, mini_batch_size):
                    yield {
                        "gen_input_ids": batch[prefix + "_gen_input_ids"][i:i + mini_batch_size],
                        "gen_attention_mask": batch[prefix + "_gen_attention_mask"][i:i + mini_batch_size], }

            user_results = []
            pbar = _iterator(self.args.train_generation_batch_size, "user")
            for mini_batch in pbar:
                result = self._batch_generate(model, mini_batch,
                                              input_ids_key="gen_input_ids",
                                              attn_mask_key="gen_attention_mask",
                                              do_sample=False if generation_samples == 1 else True,
                                              num_return_sequences=generation_samples,
                                              )
                user_results.extend(result)

        augmented_input = []
        for i in range(full_batch_size):
            element = self.train_dataset[train_data_ids[i]] | {
                "profile": user_results[i]}
            element = self.user_prompter.to_chat_example(element)
            element = self.user_prompter.totensor_multiple(element)
            augmented_input.append(element)


        augmented_input = self.data_collator(augmented_input)

        for k in augmented_input:
            if isinstance(augmented_input[k], torch.Tensor):
                augmented_input[k] = augmented_input[k].to(
                    self.accelerator.device)

        if self.state.global_step % self.args.print_out_examples == 0:
            if self.accelerator.is_main_process:
                # rich.print(self.processing_class.decode(
                #     batch["user_gen_input_ids"][0], skip_special_tokens=False))
                examples = self.processing_class.batch_decode(
                    augmented_input['multi_user_input_ids'][0:2], skip_special_tokens=False)
                examples = [e.replace(self.processing_class.pad_token, '')
                            for e in examples]
                rich.print(examples)

        return augmented_input

    

    def _efficient_forward(self,
                           model: nn.Module,
                           batch: Dict[str, Union[List, torch.LongTensor]],
                           prefix: str = "",
                           return_with_last_hidden_states: bool = False,
                           ):
        model_kwargs = {"return_dict": True,
                        "return_with_last_hidden_states": return_with_last_hidden_states,
                        }

        labels = batch[f"{prefix}_labels"]
        input_ids = batch[f"{prefix}_input_ids"]
        attention_mask = batch[f"{prefix}_attention_mask"]

        loss_mask = labels != -100
        # assert not (labels == -100).all(dim=-
        #                                 1).any().item(), "All labels are -100 in one data point"

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        # for i in range(attention_mask.size(0)):
        #     # if loss_mask is like [0, 0, 1, 1, 1, 1, 0, 0, 0],
        #     # then set the attn mask accordingly (set the back to 0)
        #     last_one_idx = torch.nonzero(loss_mask[i]).max().item()
        #     attention_mask[i][last_one_idx + 1:] = 0
        #     first_one_idx = torch.nonzero(attention_mask[i])[0].item()
        #     input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
        #     attention_mask[i] = torch.roll(
        #         attention_mask[i], shifts=-first_one_idx)
        #     loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

        # Get the first column idx that is all zeros and remove every column after that
        empty_cols = torch.sum(attention_mask, dim=0) == 0  # (seq_len, )
        first_empty_col = torch.nonzero(empty_cols)[0].item(
        ) if empty_cols.any() else attention_mask.size(1)
        input_ids = input_ids[:, :first_empty_col - 1]
        attention_mask = attention_mask[:, :first_empty_col - 1]
        loss_mask = loss_mask[:, :first_empty_col - 1]
        labels = labels[:, :first_empty_col - 1]

        # if self.args.use_logits_to_keep:
        # Compute num_logits_to_keep based on loss_mask pattern:
        # [[0, 0, 0, x, x, x, x],
        #  [0, 0, 0, x, x, x, 0]]
        #         ^ start computing logits from here ([:, -(7-3+1):])
        first_compute_index = loss_mask.nonzero(as_tuple=True)
        first_compute_index = first_compute_index[1]
        if not len(first_compute_index):  # no loss_mask
            num_logits_to_keep = loss_mask.shape[1]
        else:
            first_compute_index = first_compute_index.min()
            num_logits_to_keep = loss_mask.shape[1] - first_compute_index
            num_logits_to_keep = num_logits_to_keep.item()
        # + 1 for the first label
        model_kwargs["logits_to_keep"] = num_logits_to_keep + 1

        # Align labels with logits
        # logits:    -,  -, [x2, x3, x4, x5, x6]
        #                     ^ --------- ^       after logits[:, :-1, :]
        # labels:   [y0, y1, y2, y3, y4, y5, y6]
        #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
        # loss_mask: [0,  0,  0,  1,  1,  1,  1]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **model_kwargs,
        )
        if return_with_last_hidden_states:
            outputs, last_hidden_states = outputs

        # loss_mask[~attention_mask] = 0  # set padding tokens to 0

        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -num_logits_to_keep:]

        # labels = labels[:, 1:]
        # labels = labels[:, -num_logits_to_keep:]

        input_ids = input_ids[:, 1:]
        input_ids = input_ids[:, -num_logits_to_keep:]

        loss_mask = loss_mask[:, 1:]
        loss_mask = loss_mask[:, -num_logits_to_keep:]

        per_token_logps = selective_log_softmax(logits, input_ids)  # (B, seq)

        if return_with_last_hidden_states:
            return per_token_logps, loss_mask, last_hidden_states

        return per_token_logps, loss_mask

    def store_metrics(self, metrics, metric_key_prefix="train"):
        for key, value in metrics.items():
            self._stored_metrics[metric_key_prefix][key].append(value)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        # Create a list of keys to avoid modifying dict during iteration
        metric_key_prefixes = list(self._stored_metrics.keys())
        for metric_key_prefix in metric_key_prefixes:
            # Add averaged stored metrics to logs
            all_store_metrics = self.accelerator.gather_for_metrics(
                [self._stored_metrics[metric_key_prefix]], use_gather_object=True)

            # all_store_metrics = self._stored_metrics[train_eval]
            for key in self._stored_metrics[metric_key_prefix].keys():
                metrics = [m[key] for m in all_store_metrics]
                metrics = torch.tensor(metrics)
                metrics_mean = metrics.mean().item()
                if metric_key_prefix == "train":
                    logs[key] = metrics_mean
                else:
                    logs[f"{metric_key_prefix}_{key}"] = metrics_mean

            del self._stored_metrics[metric_key_prefix]
        return super().log(logs, *args, **kwargs)
