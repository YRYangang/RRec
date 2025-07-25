from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Union, List, Any
import datasets
import torch
from torch import nn
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import is_sagemaker_mp_enabled, logging
from trainers.GRecTrainer import GenRecTrainingArguments, GenRecTrainer

from trainers.utils import calculate_metrics
logger = logging.get_logger(__name__)

if is_sagemaker_mp_enabled():
    raise ImportError(
        "SageMaker Model Parallelism is not supported by this example.")


@dataclass
class RecPOTrainingArguments(GenRecTrainingArguments):

    epsilon_low: Optional[float] = field(
        default=0.2,
        metadata={"help": " Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=0.28,
        metadata={"help": " Epsilon value for clipping."},
    )
    reward_type: Optional[str] = field(
        default="mix",
        metadata={"help": "The reward type."},
    )
    advantage_type: Optional[str] = field(
        default="gaussian",
        metadata={
            "help": "The advantage type, either 'leave-one-out' or 'gaussian'."},
    )
    reward_ndcg_k: Optional[int] = field(
        default=1000,
        metadata={"help": "The k value for ndcg@k."},
    )
    reward_softmax_weight: Optional[float] = field(
        default=0.05,
        metadata={"help": "The weight for softmax loss."},
    )
    relabel_topk: Optional[int] = field(
        default=1,
        metadata={"help": "The k value for topk."},
    )


class RecPOTrainer(GenRecTrainer):

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],

            args: RecPOTrainingArguments,
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

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            full_dataset=full_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.args.mini_batch_size = min(self.args.per_device_train_batch_size *
                                        self.args.generation_config.num_return_sequences,
                                        self.args.mini_batch_size)

    def compute_rec_score(self,
                          model,
                          inputs,
                          similarity: Optional[torch.FloatTensor] = None,
                          train_eval='train'):
        if train_eval == 'train':
            num_samples = self.args.generation_config.num_return_sequences
            user_input_prefix = "multi_user"
        else:
            num_samples = 1
            user_input_prefix = "user"
        seq_labels = inputs['seq_labels']
        seq_input_ids = inputs['seq_input_ids']
        batch_size = seq_labels.shape[0]
        seq_labels = seq_labels.view(
            batch_size, 1, 1).expand(-1, num_samples, -1)
        seq_input_ids = seq_input_ids.view(
            batch_size, 1, -1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)

        if similarity is None:
            similarity, _ = self.compute_sim_val(model,
                                                 inputs | {
                                                     'seq_input_ids': seq_input_ids},
                                                 user_input_prefix=user_input_prefix,
                                                 )
        shaped_sim = similarity.view(
            batch_size, num_samples, -1).float()
        cutoff = self.args.reward_ndcg_k
        dcg_k, _ = calculate_metrics(shaped_sim, seq_labels, cutoff)

        ndcg = dcg_k.sum(dim=-1)  # (B, num_samples)

        sim_softmax = shaped_sim.softmax(dim=-1)

        sim_softmax = sim_softmax.gather(2, seq_labels)  # (B, num_samples, 1)
        sim_softmax = sim_softmax.squeeze(2)  # (B, num_samples)

        rewards = (1 - self.args.reward_softmax_weight) * ndcg + \
            self.args.reward_softmax_weight * sim_softmax

        if self.args.advantage_type == 'gaussian':
            if num_samples == 1:
                advantages = torch.ones_like(rewards)
            else:
                _mean = rewards.mean(dim=1, keepdim=True)
                _std = rewards.std(dim=1, keepdim=True) + 1e-8
                advantages = (rewards - _mean) / _std  # (B, num_samples)
        elif self.args.advantage_type == 'leave-one-out':
            _mean = rewards.mean(dim=1, keepdim=True)
            advantages = rewards * (1 + 1 / num_samples) - _mean
        else:
            raise NotImplementedError

        max_advantage, _ = torch.max(
            advantages, dim=-1, keepdim=True)  # 保持最后一个维度的形状
        relabel_mask = advantages == max_advantage
        topk = min(self.args.relabel_topk, advantages.shape[-1])
        _, topk_indices = advantages.topk(topk, dim=-1)
        relabel_mask = torch.zeros_like(advantages, dtype=torch.bool)
        relabel_mask.scatter_(-1, topk_indices, True)

        in_batch_labels = torch.arange(batch_size).repeat_interleave(
            num_samples).to(seq_input_ids.device)
        # e.g., num_samples=4, batch_size=8
        # 0,0,0,0,1,1,1,1...7,7,7,7
        result = {
            'advantages': advantages.view(-1),
            'relabel_mask': relabel_mask.view(-1),

            'seq_input_ids': seq_input_ids,
            'in_batch_labels': in_batch_labels,
        }

        self.store_metrics(
            {
                "softmax": sim_softmax.mean().item(),
                f"ndcg@{cutoff}": ndcg.mean().item(),
                "reward": rewards.mean().item(),
                "advantage": advantages.mean().item(),

            },
            metric_key_prefix=train_eval,
        )

        return result

    @staticmethod
    def _rec_mini_batch_iterator(inputs, macro_batch_size, mini_batch_size):
        keys = ['input_ids', 'attention_mask', 'labels']
        keys = [f'multi_user_{key}' for key in keys] + \
            ['seq_input_ids', 'advantages', 'in_batch_labels', 'relabel_mask']

        def _iterator():
            for i in range(0, macro_batch_size, mini_batch_size):
                __start = i
                __end = i + mini_batch_size
                result = {'seq_labels': inputs['seq_labels'],
                          'item_input_ids': inputs['item_input_ids'],
                          'item_attention_mask': inputs['item_attention_mask'],
                          }

                for key in keys:
                    original_in = inputs[key][__start:__end]
                    result[key] = original_in
                yield result

        return _iterator()

    def training_step(self,
                      model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]],
                      *args,
                      **kwargs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """

        inputs = self._prepare_inputs(inputs)
        eval_model = self.get_model_for_eval()
        oom = False
        try:
            with torch.no_grad():
                # if self.item_hs is None:
                #     self.generate_all_item_embeddings(eval_model)
                inputs |= self._generate_in_train(eval_model, inputs)
                inputs |= self.compute_rec_score(eval_model, inputs)
            del eval_model
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(
                    f"{self.accelerator.device}: OOM when update dset: \n {str(e)}")
                oom = True
                torch.cuda.empty_cache()
            else:
                raise e
        ooms = self.accelerator.gather_for_metrics(
            [oom], use_gather_object=True)
        if isinstance(ooms, torch.Tensor) or True in ooms:
            print("Skipping training step due to generation failure")
            return torch.tensor(0.0, requires_grad=True).to(self.args.device)

        torch.cuda.empty_cache()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        macro_batch_size = inputs["multi_user_input_ids"].shape[0]

        user_losses = []

        user_mini_batch_size = self.args.mini_batch_size
        num_user_mini_batches = macro_batch_size // user_mini_batch_size + \
            macro_batch_size % user_mini_batch_size
        user_iterator = self._rec_mini_batch_iterator(
            inputs, macro_batch_size, user_mini_batch_size)

        for i in range(num_user_mini_batches):
            try:
                _inputs = next(user_iterator)
                loss = self.batch_forward(
                    model, _inputs,
                    prefix="multi_user")
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    _progress = f"{i}/{num_user_mini_batches}"
                    print(
                        f"{self.accelerator.device}: OOM when compute loss on {_progress} mini batches")
                    torch.cuda.empty_cache()
                    loss = torch.tensor(0.0, requires_grad=True).to(
                        self.args.device)
                else:
                    raise e

            kwargs = {}
            torch.cuda.empty_cache()

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                self.accelerator.print("WARNING: ???")
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            try:
                self.accelerator.backward(loss, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    _progress = f"{i}/{num_user_mini_batches}"
                    print(
                        f"{self.accelerator.device}: OOM when backward loss on {_progress} mini batches")
                    model.zero_grad()
                else:
                    raise e

            loss = loss.detach()
            user_losses.append(loss)

        torch.cuda.empty_cache()

        # metrics = {"user_loss": 0.}

        # if len(user_losses) > 0:
        #     user_loss = torch.stack(user_losses).mean()
        #     if not torch.isnan(user_loss) and not torch.isinf(user_loss):
        #         metrics["user_loss"] = user_loss.item()
        # self.store_metrics(metrics, train_eval="train")

        # return torch.tensor(0.0, requires_grad=True).to(self.args.device)
        return torch.stack(user_losses).mean()

    def batch_forward(self, model, batch, prefix):
        advantages = batch["advantages"].unsqueeze(-1)  # (B, 1)
        _batch_size = advantages.shape[0]
        relabel_mask = batch["relabel_mask"]

        per_token_logps, loss_mask, last_hidden_states = self._efficient_forward(
            model,
            batch,
            prefix,
            return_with_last_hidden_states=True,
        )
        coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
        coef_2 = torch.clamp(coef_1,
                             1 - self.args.epsilon_low,
                             1 + self.args.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(-1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(-1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        similarity, _ = self.compute_sim_train(model=model, inputs=batch,
                                               user_h=last_hidden_states,
                                               user_input_prefix=prefix,
                                               )
        labels = batch['in_batch_labels']
        num_processes = self.accelerator.num_processes
        if self.args.gather_negs_across_processes and num_processes > 1:
            # num_samples=4, batch_size=8, mini_batch_size=5
            # {0,0,0,0,1},{1,1,1,2,2}...{6,6,6,7,7},{7,7}
            # if gather negs, with 3 cudas, we have 3*8 items in total
            # num_labels in a mini batch now is 5*3
            # {0,0,0,0,1, 0+8,0+8,0+8,0+8,1+8, 0+16,0+16,0+16,0+16,1+16}
            _macro_batch_size = batch['item_input_ids'].shape[0]
            # turn {0,0} -> {0,0,0+8,0+8,0+16,0+16}
            labels = torch.cat([
                labels + i * _macro_batch_size
                for i in range(num_processes)
            ], dim=0)

            _gathered_advantages = self._gather_and_cat(advantages)
            relabel_mask = self._gather_and_cat(relabel_mask)

            # assert labels.shape[0] == _batch_size * num_processes
            # assert labels.shape[0] == similarity.shape[0]
        else:
            _gathered_advantages = advantages

        _gathered_advantages = _gathered_advantages * relabel_mask.float().unsqueeze(
            -1)
        assert _gathered_advantages.shape[0] == labels.shape[
            0], f"{_gathered_advantages.shape} {labels.shape}"
        assert _gathered_advantages.shape[1] == 1, f"{_gathered_advantages.shape}"

        similarity = similarity.softmax(dim=-1)
        probs = similarity.gather(dim=-1, index=labels.unsqueeze(-1))
        coef_1 = probs / probs.detach()
        coef_2 = torch.clamp(coef_1,
                             1 - self.args.epsilon_low,
                             1 + self.args.epsilon_high)
        item_loss1 = coef_1 * _gathered_advantages.unsqueeze(-1)
        item_loss2 = coef_2 * _gathered_advantages.unsqueeze(-1)
        item_loss = -torch.min(item_loss1, item_loss2)
        item_loss = (item_loss).sum(dim=-1).mean()
        tokens_loss = (per_token_loss * loss_mask).sum()
        tokens_loss /= loss_mask.sum()
        tokens_loss /= _batch_size
        return item_loss + tokens_loss

    def compute_sim_train(self, model, inputs,
                          user_input_prefix: str = "user",
                          item_input_prefix: str = "item",
                          user_h: Optional[torch.FloatTensor] = None,
                          item_h: Optional[torch.FloatTensor] = None,
                          ):
        batch_size = inputs[f"{user_input_prefix}_input_ids"].shape[0]
        seq_labels = inputs["seq_labels"]  # (B,)
        # seq_input_ids = inputs["seq_input_ids"]  # (B, seq)
        if user_h is None:
            user_h = model(
                attention_mask=inputs[f"{user_input_prefix}_attention_mask"],
                input_ids=inputs[f"{user_input_prefix}_input_ids"],
                return_causal_output=False,
                return_with_last_hidden_states=True,)

        if item_h is None:
            item_h = model(
                attention_mask=inputs[f"{item_input_prefix}_attention_mask"],
                input_ids=inputs[f"{item_input_prefix}_input_ids"],
                return_causal_output=False,
                return_with_last_hidden_states=True,)

        num_processes = self.accelerator.num_processes
        if num_processes > 1 and self.args.gather_negs_across_processes:
            # copy and detach the item_h
            tmp_item_h = item_h.detach()
            # (4 * B_2, hidden_size)
            gathered_item_h = self.accelerator.gather_for_metrics([tmp_item_h],
                                                                  use_gather_object=True)
            tmp_user_h = user_h.detach()
            # (4 * B, hidden_size)
            gathered_user_h = self.accelerator.gather_for_metrics([tmp_user_h],
                                                                  use_gather_object=True)

            gathered_seq_labels = self.accelerator.gather_for_metrics([seq_labels],
                                                                      use_gather_object=True)

            current_process = self.accelerator.process_index
            indices_to_take = [i for i in range(
                num_processes) if i != current_process]

            item_h = torch.cat(
                [item_h] + [gathered_item_h[i].to(item_h.device) for i in indices_to_take], dim=0)
            item_h = item_h.view(-1, item_h.size(-1))
            user_h = torch.cat(
                [user_h] + [gathered_user_h[i].to(item_h.device) for i in indices_to_take], dim=0)
            user_h = user_h.view(-1, user_h.size(-1))

            seq_labels = torch.cat(
                [seq_labels] + [gathered_seq_labels[i].to(item_h.device) for i in indices_to_take], dim=0)
            seq_labels = seq_labels.view(-1)

            labels = torch.arange(
                batch_size * num_processes, device=user_h.device)

        else:
            labels = torch.arange(batch_size, device=user_h.device)  # (B,)

        self.item_hs[seq_labels] = item_h.detach().to(self.item_hs.dtype)

        similarity = self.similarity(user_h, item_h)

        return similarity, labels

    def compute_sim_val(self, model, inputs,
                        user_input_prefix: str = "user",
                        user_h: Optional[torch.FloatTensor] = None,
                        item_h: Optional[torch.FloatTensor] = None,
                        ):

        if item_h is None:

            if self.item_hs is None:
                self._generate_item_embeddings(model)
            item_h = self.item_hs

        if user_h is None:
            user_h = model(
                attention_mask=inputs[f"{user_input_prefix}_attention_mask"],
                input_ids=inputs[f"{user_input_prefix}_input_ids"],
                return_causal_output=False,
                return_with_last_hidden_states=True,
            )
        similarity = self.similarity(user_h, item_h)
        seq_input_ids = inputs["seq_input_ids"]
        assert seq_input_ids.shape[0] == user_h.shape[
            0], f"{seq_input_ids.shape[0]} != {user_h.shape[0]}"
        interacted_mask = torch.zeros(similarity.size(0), similarity.size(1),
                                      dtype=torch.bool, device=similarity.device)
        interacted_mask.scatter_(1, seq_input_ids, 1)
        similarity[interacted_mask] = torch.finfo(similarity.dtype).min

        return similarity, inputs.get('seq_labels', None)

    def compute_loss(self, model, inputs,
                     return_outputs=False,
                     **loss_kwargs):
        """
        This is the method that will only be called in eval mode (since we manually overwrite `training_step` method)
        We will compute the similarity between user and item, and the RL reward, advantage.
        """
        assert not model.training
        similarity, _ = self.compute_sim_val(model, inputs)
        loss = torch.tensor(0.0).to(similarity.device)

        if return_outputs:
            outputs = CausalLMOutput(
                loss=None,
                logits=similarity,
                hidden_states=None,
                attentions=None,
            )

            self.compute_rec_score(
                model,
                inputs,
                similarity=outputs.logits,
                train_eval='eval')

        return loss, outputs
