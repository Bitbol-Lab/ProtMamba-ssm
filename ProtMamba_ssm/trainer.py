from transformers import Trainer, TrainerCallback
from .utils import *
import re
import torch
import os

class MambaTrainer(Trainer):
    """
    Base HuggingFace Trainer used for training.
    
    from https://github.com/havenhq/mamba-chat/blob/main/trainer/mamba_trainer.py"""
    def __init__(self, compute_only_fim_loss, **kwargs,):
        super().__init__(**kwargs)
        self.compute_only_fim_loss = compute_only_fim_loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        if "seq_position_ids" in inputs and "position_ids" in inputs:
            position_ids = inputs.pop("position_ids")
            seq_position_ids = inputs.pop("seq_position_ids")
            output = model(input_ids, position_ids=position_ids, seq_position_ids=seq_position_ids)
        elif "position_ids" in inputs:
            position_ids = inputs.pop("position_ids")
            output = model(input_ids, position_ids=position_ids)
        else:
            output = model(input_ids)
        lm_logits = output.logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        if self.compute_only_fim_loss:
            # start and end tokens
            is_cls_tokens = (labels == AA_TO_ID["<cls>"])
            is_eos_tokens = (labels == AA_TO_ID["<eos>"])
            bool_fim = find_fim_indices(is_cls_tokens, is_eos_tokens)
            # include also the cls token
            bool_fim = bool_fim | is_cls_tokens
            inds = torch.where(bool_fim)
            lm_loss = loss_fct(shift_logits[inds[0], inds[1], :], labels[bool_fim])
        else:
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return (lm_loss, output) if return_outputs else lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder, max_steps=None):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    
    max_steps = max_steps if max_steps is not None else float("inf")
    # func = lambda x: int(_re_checkpoint.search(x).groups()[0])
    def func(x):
        num = int(_re_checkpoint.search(x).groups()[0])
        return num if num < max_steps else -1
    return os.path.join(folder, max(checkpoints, key=func))

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, train_path, config=None):
        self.step_counter = 0
        self.best_loss = None
        self.train_path = train_path
        self.patience = config["patience"]
        self.metric_name = config["early_stopping_metric"]
        self.checkpoint_path = None
        self.should_restart = False
        self.eval_steps = config["eval_steps"]
        self.loss_increase_factor = config["loss_increase_factor"]
    
    def get_checkpoint_path(self, max_steps):
        last_checkpoint = None
        if os.path.exists(self.train_path):
            last_checkpoint = get_last_checkpoint(self.train_path, max_steps)
            if last_checkpoint is None:
                print("No checkpoint found, starting training from scratch.")
            else:
                print(f"Max checkpoint allowed: {max_steps}, restarting from {last_checkpoint}.")
        return last_checkpoint

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        if self.metric_name in metrics:
            if self.best_loss is None:
                self.best_loss = metrics[self.metric_name]
            elif self.best_loss*self.loss_increase_factor < metrics[self.metric_name]:
                self.step_counter += 1
                if self.step_counter >= self.patience:
                    checkpoint_path = self.get_checkpoint_path(max_steps=(state.global_step-self.patience*self.eval_steps))
                    control.should_training_stop = True
                    self.checkpoint_path = checkpoint_path
                    self.should_restart = True 
            else:
                self.step_counter = 0
                self.best_loss = min(self.best_loss, metrics[self.metric_name])
                self.should_restart = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.step_counter = 0
        self.best_loss = None
        self.should_restart = False
