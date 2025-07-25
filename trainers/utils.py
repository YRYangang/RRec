
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EvalPrediction


class MetricUpdater:
    def __init__(self, ks=None):
        self.metric_collection = None

        if ks is None:
            ks = [5, 10, 20, 50]
        self.ks = ks
        self.max_k = max(self.ks)

        # Initialize metric storage
        self._init_metrics()

    def _init_metrics(self):
        self.ndcg_metric = {k: 0. for k in self.ks}
        self.hr_metric = {k: 0. for k in self.ks}
        self.sample_count = 0

    def update(self, logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor):
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Check the input
        valid = _check_valid_input(logits, labels)
        if not valid:
            return

        dcg_k, hit = calculate_metrics(logits, labels, self.max_k)
        # Accumulate NDCG and Hit Rate metrics
        for k in self.ks:
            # NDCG@k
            ndcg_k = dcg_k[:, :k].sum(dim=1)
            # Sum over batch without averaging
            self.ndcg_metric[k] += ndcg_k.sum().item()

            # Hit@k
            hits_k = hit[:, :k].sum(dim=1)
            self.hr_metric[k] += hits_k.sum().item()  # Sum of hits over batch

        self.sample_count += labels.size(0)

    def compute(self) -> Dict[str, float]:
        result = {}
        sample_count = self.sample_count
        for k in self.ndcg_metric:
            result[f"ndcg@{k}"] = self.ndcg_metric[k] / sample_count
        for k in self.hr_metric:
            result[f"hit_rate@{k}"] = self.hr_metric[k] / sample_count
        self._init_metrics()

        return result


def get_compute_metrics(metric_updater: MetricUpdater, num_negatives: Optional[int] = None) -> Callable[
        [EvalPrediction, bool], Dict[str, float]]:
    def compute_metrics(eval_pred: EvalPrediction, compute_result=False) -> Dict[str, float]:
        logits = eval_pred.predictions  # (B, seq, num_items)
        labels = eval_pred.label_ids  # (B, seq)

        # exclude batches with no valid samples
        # valid_batch_index = labels.sum(-1) != -100 * labels.shape[-1]
        if labels.shape[0] != logits.shape[0]:
            device_id = logits.device.index
            logits = logits[device_id * labels.shape[0]: (device_id + 1) * labels.shape[0]]
            assert labels.shape[0] == logits.shape[0], f"{labels.shape[0]} != {logits.shape[0]}"

        labels = labels.view(-1)
        idx = labels.ne(-100)
        labels = labels[idx]
        logits = logits.view(-1, logits.size(-1))[idx]

        # logits: (B, num_items), labels: (B,)
        if num_negatives is not None and num_negatives > 0:
            sampled_labels, sampled_logits = _negative_sampling(labels, logits)
            metric_updater.update(logits=sampled_logits, labels=sampled_labels)
        else:
            metric_updater.update(logits=logits, labels=labels)

        if compute_result:
            result = metric_updater.compute()
            return result

    def _negative_sampling(labels, logits):
        B, num_items = logits.shape
        sampling_prob = torch.ones(
            (B, num_items), dtype=torch.float, device=labels.device)
        sampling_prob[torch.arange(B), labels] = 0  # 将正样本位置的概率设为0
        negative_items = torch.multinomial(
            sampling_prob, num_samples=num_negatives, replacement=False)
        sampled_items = torch.cat([labels.view(-1, 1), negative_items], dim=-1)
        sampled_logits = torch.gather(logits, dim=-1, index=sampled_items)
        sampled_labels = torch.zeros(B, dtype=torch.long, device=labels.device)
        return sampled_labels, sampled_logits

    return compute_metrics



def _check_valid_input(logits, labels) -> bool:
    # check if empty
    if not logits.numel() or not labels.numel():
        return False

    if logits.size(0) != labels.size(0):
        raise ValueError(
            f"Batch dimension of logits and labels must be the same. Got logits: {logits.size(0)}, labels: {labels.size(0)}")
    # check nan
    if torch.isnan(logits).any():
        raise ValueError("logits contains nan")

    if labels.max().item() >= logits.shape[-1]:
        raise ValueError(
            f"labels contain values greater than the number of classes. Got max label: {labels.max().item()}, num_classes: "
            f"{logits.size(-1)}")

    return True


def calculate_metrics(
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    cutoff: int,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """
    Calculate the DCG (Discounted Cumulative Gain) for a batch of predictions and labels.

    Args:
        logits (torch.FloatTensor): The predicted scores for each item, shape: (*, num_items).
        labels (torch.IntTensor): The ground truth labels for each item: (*) or (*, 1)
        cutoff (int): The cutoff value for NDCG calculation.

    Returns:
        torch.FloatTensor: The DCG values for each item in the batch, shape: (*, cutoff).
        torch.BoolTensor: The hit values for each item in the batch, shape: (*, cutoff).

    """
    # labels shape must equal to preds shape except the last dimension
    if len(logits.shape) == len(labels.shape) + 1:
        labels = labels.unsqueeze(-1)
    else:
        assert len(logits.shape) == len(labels.shape), f"{len(logits.shape)} != {len(labels.shape)}"
        assert logits.shape[:-1] == labels.shape[:-1], f"{logits.shape[:-1]} != {labels.shape[:-1]}"
        assert labels.shape[-1] == 1, f"{labels.shape[-1]} != 1"
    _shape = labels.shape[:-1] + (cutoff,)
    labels = labels.expand(_shape) # (*, cutoff)
    
    preds = logits.topk(cutoff, dim=-1).indices
    hit = (preds.squeeze(-1) == labels)

    discount = torch.log2(torch.arange(2, cutoff + 2,
                                       dtype=torch.float32,
                                       device=labels.device))
    dcg = (1.0 / discount)  # (cutoff,)
    dcg = torch.where(hit, dcg, 0)  # (*, cutoff)

    return dcg, hit



class Similarity:
    """
    Dot product or cosine similarity
    """

    def __init__(self, config):
        similarity_type = config.similarity_type
        if similarity_type == "cosine":
            self.forward_func = self.forward_cos
        elif similarity_type == "dot":
            self.forward_func = self.forward_dot
        elif similarity_type == "L2":
            self.forward_func = self.forward_l2
        else:
            raise NotImplementedError(
                f"Similarity type {similarity_type} not implemented")
        self.similarity_type = config.similarity_type
        self.temp = config.similarity_temperature
        self.do_normalize = config.similarity_normalization

    def forward_dot(self, x, y):
        return torch.matmul(x, y.t())

    def forward_cos(self, x, y):
        return nn.CosineSimilarity(dim=-1)(x, y)

    def forward_l2(self, x, y):
        return -torch.norm(x.unsqueeze(1) - y.unsqueeze(0), p=2, dim=-1)

    def forward(self, x, y):
        # check if dtypes are the same
        if not x.dtype == y.dtype:
            x = x.to(y.dtype)
        if self.do_normalize:
            x = nn.functional.normalize(x, p=2, dim=-1)
            y = nn.functional.normalize(y, p=2, dim=-1)
        result = self.forward_func(x, y) / self.temp

        assert result.shape == (x.shape[0], y.shape[0])
        return result

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "qwen" in str(tokenizer.__class__).lower() or "qwen" in model_name.lower():
        tokenizer.__setattr__('generation_prompt', "<|im_start|>assistant\n")
        tokenizer.__setattr__('generation_end', "<|im_end|>")


    if "gemma" in str(tokenizer.__class__).lower() or "gemma" in model_name.lower():
        tokenizer.__setattr__('generation_prompt', "<start_of_turn>model\n")
        tokenizer.__setattr__('generation_end', "<end_of_turn>")

    tokenizer.padding_side = "right"

    return tokenizer
    