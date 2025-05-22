
import os
import datasets
import rich
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback, GenerationConfig
from data_collators.data_collator import RRecDataCollator as DataCollator
from paths import model_names
from trainers.utils import get_compute_metrics, get_tokenizer, MetricUpdater
from trainers.RecPOTrainer import (
    RecPOTrainer,
    RecPOTrainingArguments)


def train(
        output_dir="../checkpoints",

        run_name: str = "dpo-6epoch-debug-v2",

        train_batch_size: int = 4,
        eval_batch_size: int = 32,

        train_generation_batch_size=16,
        test_generation_batch_size=32,

        item_emb_batch_size: int = 128,
        warmup_steps: int = 32,

        eval_freq=8,
        early_stopping_patience=8,
        eval_on_start: bool = True,
        logging_dir="./logs",
        gradient_accumulation_steps: int = 1,

        num_train_epochs: int = 10,
        learning_rate: float = 1e-5,


        cleanup_previous_checkpoints=False,
        
        dataset_category: str = "",
        dataset_dir="",
        
        use_lora=True,
        seed=42,
        model = 'gemma',

        resume_from_checkpoint: bool = False,
        window_size: int = 20,
        gather_negs_across_processes=True,
        max_new_tokens=300,
        group_size=2,
        lr_scheduler_type='constant',
        use_vllm=True,
        **kwargs,
):
    trainer_extra_kwargs = dict()
    lora_kwargs = dict()
    for k in kwargs:
        if k.startswith('trainer'):
            trainer_extra_kwargs[k.replace('trainer_', '')] = kwargs[k]
        else:
            lora_kwargs[k] = kwargs[k]
    del kwargs

    datasets.disable_progress_bars()
    if model == 'gemma':
        model_name = model_names["Gemma-2-2b-it"]
        from models.gemma_models import (Gemma2RRecCasualLM as ModelClass,
                                         Gemma2RRecConfig as ConfigClass)
    elif model == 'qwen':
        model_name = model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import (Qwen2RRecCasualLM as ModelClass,
                                        Qwen2RRecConfig as ConfigClass)
    else:
        raise NotImplementedError
    output_dir = os.path.join(output_dir, run_name)

    accelerator = Accelerator()
    rich.print(accelerator.deepspeed_plugin)

    if accelerator.is_main_process:
        rich.print("Arguments: ", locals())

    ################## set dataset ##################

    dset = datasets.load_from_disk(dataset_dir)


    tokenizer = get_tokenizer(model_name)

    emb_token = '<answer>'
    emb_end_token = '</answer>'

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    ################### set model ###################

    base_model = ModelClass.from_pretrained(model_name,
                                            torch_dtype=torch.bfloat16,
                                            # attn_implementation="flash_attention_2",
                                            device_map={
                                                "": accelerator.process_index},
                                            config=config)

    ################### set generation ###################

    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    gen_config.top_k = 200
    gen_config.top_p = 1.0
    gen_config.temperature = 1.5

    ################################################################


    peft_config_dict = {
        "inference_mode": False,

        "target_modules": [
            'k_proj', 'v_proj', 'q_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    peft_config_dict.update(lora_kwargs)

    if use_lora:
        lora_cfg = {"r": 4, "lora_alpha": 128, }
        lora_cfg.update(peft_config_dict)
        peft_config = LoraConfig(**lora_cfg)
        if accelerator.is_main_process:
            rich.print(peft_config)
        base_model = get_peft_model(base_model, peft_config)
    else:
        if accelerator.is_main_process:
            rich.print("No PEFT applied, training the base model")

    # base_model.enable_input_require_grads()
    ################### set trainer ###################
    # calculate steps required for half an epoch
    eval_steps = len(dset['train']) / (train_batch_size *
                                       gradient_accumulation_steps * 3)
    eval_steps = eval_steps // eval_freq

    training_args = RecPOTrainingArguments(
        seed=seed,

        item_emb_batch_size=item_emb_batch_size,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=1.,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,

        save_strategy="steps",
        save_steps=eval_steps,
        save_only_model=False,
        save_total_limit=10,
        load_best_model_at_end=True,

        eval_strategy="steps",
        eval_steps=eval_steps,

        bf16_full_eval=True,
        per_device_eval_batch_size=eval_batch_size,
        metric_for_best_model='ndcg@20',
        eval_on_start=eval_on_start,
        batch_eval_metrics=True,

        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        report_to='none',
        run_name=run_name,
        logging_dir=logging_dir,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        gather_negs_across_processes=gather_negs_across_processes,
        generation_config=gen_config,

        train_generation_batch_size=train_generation_batch_size,
        test_generation_batch_size=test_generation_batch_size,

        dataset_window_size=window_size,
        dataset_category=dataset_category,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,
        **trainer_extra_kwargs,

    )
    metric_updater = MetricUpdater(ks=[5, 10, 20])

    trainer = RecPOTrainer(
        model=base_model,
        compute_metrics=get_compute_metrics(metric_updater, ),
        data_collator=DataCollator(tokenizer=tokenizer,
                                   return_tensors="pt"),

        full_dataset=dset,

        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience)],
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if cleanup_previous_checkpoints:
        os.system(f"rm -rf {output_dir}/checkpoint-*")
        print(f"Removed previous checkpoints in {output_dir}")

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(train)
