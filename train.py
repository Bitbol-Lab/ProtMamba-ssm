
# Modules
from ProtMamba_ssm.modules import *
# Trainer
from ProtMamba_ssm.trainer import *
# Dataloaders
from ProtMamba_ssm.dataloaders import *
# Utils
from ProtMamba_ssm.utils import *
from transformers import TrainingArguments, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup

import yaml
import torch
import numpy as np
from transformers.integrations import TensorBoardCallback
from torch.optim import AdamW

# List of available models
_mamba_model = {"none": MambaLMHeadModelSafe, "1d": MambaLMHeadModelwithPosids, "2d": MambaLMHeadModelwith2DPosids}


def run(config, namedir=None, finetune_model_path=None, restart_optimizer_and_scheduler=False):
    r"""Run the training/finetuning loop.

    Args:
        config (dict): dictionary with the configuration parameters.
        namedir (str, optional): name of the directory where the model will be saved. If None, the name will be taken from the config file.
        finetune_model_path (str, optional): path to the model to be finetuned. If None, a new model will be created.
    """
    if namedir is None:
        namedir = config["namedir"]
    # Load Dataset
    full_dataset = Uniclust30_Dataset(filename=config["train_dataset_path"],
                                      filepath=config["data_dir"],
                                      sample=config["sample_sequences"],
                                      max_msa_len=config["max_msa_len"],
                                      reverse=config["reverse"],
                                      seed=config["seed_sequence_sampling"],
                                      troubleshoot=False,
                                      fim_strategy=config["fim_strategy"],
                                      always_mask=config["always_mask"],
                                      max_position_embeddings=config["max_position_embeddings"],
                                      max_seq_position_embeddings=config["max_seq_position_embeddings"],
                                      add_position_ids=config["add_position_ids"])

    assert len(AA_TO_ID) == config[
        "vocab_size"], f"Vocab size in the config file does not match the one in the code. I should be {len(AA_TO_ID)}"

    # Split dataset in train, validation and test
    gen = torch.Generator()
    gen.manual_seed(config["seed_datasets"])
    np_gen = np.random.default_rng(seed=config["seed_datasets"])
    len_full_dataset = len(full_dataset)
    len_val = config["size_validation"]  # len_full_dataset - len_train
    len_train = len_full_dataset - len_val  # int(0.99 * len_full_dataset)
    assert len_val < len_full_dataset, "Validation set is larger than the full dataset"
    assert len_val % config["batch_size"] == 0, "Validation set size must be a multiple of the batch size"
    print(f"Train: {len_train} samples, Validation: {len_val} samples")
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [len_train, len_val], generator=gen)
    eval_train_dataset = torch.utils.data.Subset(train_dataset,
                                                 np_gen.choice(len(train_dataset), len(eval_dataset)))
    # Create data collator for batched training
    data_collator = DataCollatorForUniclust30Dataset()

    # Configure Mamba model
    Mamba = _mamba_model[config["add_position_ids"]]
    if config["dtype"] == "float32":
        dtype = torch.float32
        print("Using float32")
    elif config["dtype"] == "bfloat16":
        dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        raise ValueError("dtype must be either float32 or bfloat16")
    if finetune_model_path is not None:
        # Load model for finetuning
        model = load_model(finetune_model_path, device="cuda", model_class=Mamba, dtype=dtype,
                           checkpoint_mixer=config["checkpoint_mixer"])
    else:
        # Create model for training
        mamba_config = MambaConfig(d_model=config["d_model"],
                                   n_layer=config["n_layer"],
                                   vocab_size=config["vocab_size"],
                                   residual_in_fp32=config["residual_in_fp32"])
        model = Mamba(mamba_config, dtype=dtype, checkpoint_mixer=config["checkpoint_mixer"])

    # Print model information
    print_number_of_parameters(model)
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    eff_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
    nr_gpus = torch.cuda.device_count()
    eff_batch_size *= nr_gpus
    print(f"Effective batch size: {eff_batch_size}")
    print(
        f"Steps per training epoch: {len(train_dataset) // config['batch_size']}, eff. steps: {len(train_dataset) // eff_batch_size}")
    print(f"Steps per evaluation epoch: {len(eval_dataset) // config['batch_size']}")
    print(f"Max MSA length: {config['max_msa_len']}")
    ev_epochs = round(config['eval_steps'] * config["batch_size"] / len(train_dataset), 3)
    print(
        f"Evaluation every {config['eval_steps']} steps, i.e. {ev_epochs} epochs. Effectively every {config['eval_steps'] * config['gradient_accumulation_steps']} steps, i.e. {ev_epochs * config['gradient_accumulation_steps']} epochs.")

    # Training callbacks
    es_callback = EarlyStoppingCallback(train_path=config["output_dir"] + namedir, config=config)
    callbacks = [TensorBoardCallback(), es_callback]

    # Optimizer and Schedulers
    optimizer = AdamW(model.parameters(),
                      lr=config["learning_rate"],
                      betas=(config["beta1"], config["beta2"]),
                      weight_decay=config["weight_decay"])
    if config["scheduler"] == "cosine":
        print("Using cosine scheduler")
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config["warmup_steps"],
                                                    num_training_steps=config["num_epochs"] * len(
                                                        train_dataset) // eff_batch_size)
    if config["scheduler"] == "cosine-restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=config["warmup_steps"],
                                                                       num_training_steps=config["num_epochs"] * len(
                                                                           train_dataset) // eff_batch_size,
                                                                       num_cycles=config["num_cycles"])
    elif config["scheduler"] == "constant":
        print("Using constant scheduler with warmup")
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"])
    else:
        raise ValueError("Scheduler must be either cosine or constant")

    if finetune_model_path is not None:
        if not restart_optimizer_and_scheduler:
            optimizer.load_state_dict(torch.load(finetune_model_path + "/optimizer.pt"))
            scheduler.load_state_dict(torch.load(finetune_model_path + "/scheduler.pt"))

    # Find checkpoint if available
    last_checkpoint = None
    if finetune_model_path is None:
        if os.path.exists(config["output_dir"] + namedir):
            last_checkpoint = get_last_checkpoint(config["output_dir"] + namedir)
            if last_checkpoint is None:
                print("No checkpoint found, starting training from scratch.")
            else:
                print(f"Resuming training from the last checkpoint: {last_checkpoint}")
    if config["checkpoint_mixer"]:
        print("Using gradient checkpointing")
    # Create model trainer
    trainer = MambaTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset={"valid": eval_dataset, "train": eval_train_dataset},
        optimizers=(optimizer, scheduler),
        args=TrainingArguments(
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            eval_accumulation_steps=config["eval_accumulation_steps"],
            eval_strategy="steps",
            max_grad_norm=config["max_grad_norm"],
            bf16=config["dtype"] == "bfloat16",
            dataloader_num_workers=1,
            logging_steps=config["logging_steps"],
            eval_steps=config["eval_steps"],
            save_steps=config["save_steps"],
            output_dir=config["output_dir"] + namedir,
            logging_dir=config["output_dir"] + namedir,
            overwrite_output_dir=False,
            push_to_hub=False,
            label_names=["labels"],
        ),
        compute_only_fim_loss=config["compute_only_fim_loss"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks)
    assert trainer.args._n_gpu == nr_gpus, "Number of gpus used is not the same as the number of gpus available"
    if trainer.compute_only_fim_loss:
        print("Computing only FIM loss for training")
    # Train model
    while True:
        if last_checkpoint is None and trainer.state.global_step == 0:
            eval_results = trainer.evaluate()
            print(f">>> Initial Perplexity: {eval_results['eval_valid_perplexity/batch']:.2f}")
        else:
            print(f"Resuming training from the last checkpoint: {last_checkpoint}")
        # Train
        trainer.train(resume_from_checkpoint=last_checkpoint)
        # Break training when the number of epochs is reached
        if not es_callback.should_restart or trainer.state.epoch >= config["num_epochs"]:
            eval_results = trainer.evaluate()
            print(f">>> Final Perplexity: {eval_results['eval_valid_perplexity/batch']:.2f}")
            break
        # If the training was interrupted because of a loss spike, restart from the last checkpoint
        last_checkpoint = es_callback.checkpoint_path

    return trainer


if __name__ == "__main__":
    use_one_gpu = "0"
    if int(use_one_gpu) >= 0:
        print(f"Using gpu {use_one_gpu}")
    print("Number of gpus used: ", torch.cuda.device_count())

    with open("configs/default_config.yaml", "r") as file:
        defaultconfig = yaml.safe_load(file)

    namedir = "test"
    trainer = run(defaultconfig, namedir)

    run(defaultconfig, namedir=None, finetune_model_path=None, restart_optimizer_and_scheduler=False)