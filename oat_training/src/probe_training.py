import json
import os
import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from tqdm.auto import tqdm
import wandb

from .attacks import train_attack
from .utils import convert_seconds_to_time_str, get_valid_token_mask, calculate_forward_flops, calculate_backward_flops, get_model_size


class Probe(nn.Module):
    # Base class for all probes

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # assert x.dim() == 3, "Input must be of shape (batch_size, seq_len, d_model)"
        raise NotImplementedError

    def compute_loss(self, acts, labels, mask=None):
        # acts should be of shape (d1, d2, ..., dn, d_model)
        # labels should be of shape (d1, d2, ..., dn)
        # where d1, d2, ..., dn are the batch dimensions

        logits = self.forward(acts)

        # Handle masking
        if mask is not None:
            # Ensure mask shape matches logits shape
            if mask.shape != logits.shape:
                # If mask is flattened, reshape it to match logits
                mask = mask.view(logits.shape)

            # Apply mask
            logits = logits[mask]
            labels = labels[mask]

        # Compute BCE loss
        labels = labels.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

    def predict(self, x):
        # x should be of shape (d1, d2, ..., dn, d_model)
        # should return a tensor of shape (d1, d2, ..., dn)
        # All returned values should be between 0 and 1
        return torch.sigmoid(self.forward(x))


def initialize_probes_and_optimizers(
    layers, create_probe_fn, lr, device, pretrained_probes=None
):
    # Initialize probes and their corresponding optimizers for each layer
    if pretrained_probes is not None:
        print("Using pretrained probes...")
        probes = pretrained_probes
    else:
        probes = {layer: create_probe_fn() for layer in layers}
    optimizers = {
        layer: torch.optim.AdamW(probe.parameters(), lr=lr)
        for layer, probe in probes.items()
    }
    return probes, optimizers


def train_layer(
    layer,
    probe,
    optimizer,
    pos_activations,
    neg_activations,
    n_epochs,
    batch_size,
    n_grad_accum,
    device,
    using_memmap,
    clip_grad_norm=1.0,
):
    # Train a probe on the activations at a specific layer
    probe.to(device)
    n_examples = min(len(pos_activations), len(neg_activations))
    total_losses = []

    for epoch in tqdm(range(n_epochs)):

        # Shuffle the activations every epoch
        epoch_loss = 0
        shuffle_indices = np.random.permutation(n_examples)
        pos_activations_shuffled = pos_activations[shuffle_indices]
        neg_activations_shuffled = neg_activations[shuffle_indices]

        for i in range(0, n_examples, batch_size):

            # Drop last batch if it is smaller than batch_size
            if i + batch_size > n_examples:
                break

            # Train the probe on the batch of activations
            with torch.autocast(device_type=device):
                probe.train()

                # Load the batch onto the device, and create masks for zero padding
                if not using_memmap:
                    pos_batch = pos_activations_shuffled[i : i + batch_size].to(device)
                    neg_batch = neg_activations_shuffled[i : i + batch_size].to(device)
                else:
                    pos_batch = torch.from_numpy(
                        pos_activations_shuffled[i : i + batch_size]
                    ).to(device)
                    neg_batch = torch.from_numpy(
                        neg_activations_shuffled[i : i + batch_size]
                    ).to(device)
                zero_mask_pos = torch.all(pos_batch == 0, dim=-1).view(-1).to(device)
                zero_mask_neg = torch.all(neg_batch == 0, dim=-1).view(-1).to(device)

                # Forward pass through the probe, and compute the loss
                pos_targets = torch.ones_like(pos_batch[..., 0], device=device)
                neg_targets = torch.zeros_like(neg_batch[..., 0], device=device)

                loss_pos = probe.compute_loss(
                    pos_batch, pos_targets, mask=~zero_mask_pos
                )
                loss_neg = probe.compute_loss(
                    neg_batch, neg_targets, mask=~zero_mask_neg
                )

                loss = (loss_pos + loss_neg) / n_grad_accum

            # Backward pass and optimization step
            loss.backward()
            epoch_loss += loss.item() * n_grad_accum

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_grad_norm)

            if (i // batch_size + 1) % n_grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Perform an extra optimization step if the number of examples is not divisible by batch_size
        if (n_examples // batch_size) % n_grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        total_losses.append(epoch_loss)

    probe.to("cpu")
    return layer, probe, total_losses


def cache_activations(encoder, examples, batch_size, max_length, cache_dir, **kwargs):
    # Cache activations for a set of examples using the encoder
    initial_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"  # Use right padding
    activations = encoder.get_model_residual_acts(
        examples,
        batch_size=batch_size,
        max_length=max_length,
        use_memmap=cache_dir,
        **kwargs,
    )
    encoder.tokenizer.padding_side = initial_padding_side
    return activations


def train_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    use_parallelism=False,
    lr=1e-3,
    max_length=1024,
    n_epochs=10,
    batch_size=16,
    n_grad_accum=1,
    device="cuda",
    cache_activations_save_path=None,
    pretrained_probes=None,
    **kwargs,
):
    # Main function to train probes for all specified layers

    # Check if the cache file exists and a save path is provided
    if cache_activations_save_path is not None and os.path.exists(
        cache_activations_save_path
    ):
        print(f"Loading cached activations from {cache_activations_save_path}")

        positive_metadata_file = os.path.join(
            cache_activations_save_path, "positive_examples_metadata.json"
        )
        negative_metadata_file = os.path.join(
            cache_activations_save_path, "negative_examples_metadata.json"
        )

        # Load the memmaps for the positive examples
        positive_activations = []
        with open(positive_metadata_file, "r") as f:
            positive_metadata = json.load(f)
            for layer in range(positive_metadata["num_layers"]):
                pos_file = os.path.join(
                    cache_activations_save_path,
                    f"positive_examples_residual_act_layer_{layer}.dat",
                )
                pos_memmap = np.memmap(
                    pos_file,
                    dtype=positive_metadata["dtype"],
                    mode="r",
                    shape=tuple(positive_metadata["shape"]),
                )
                positive_activations.append(pos_memmap)

        # Load the memmaps for the negative examples
        negative_activations = []
        with open(negative_metadata_file, "r") as f:
            negative_metadata = json.load(f)
            for layer in range(negative_metadata["num_layers"]):
                neg_file = os.path.join(
                    cache_activations_save_path,
                    f"negative_examples_residual_act_layer_{layer}.dat",
                )
                neg_memmap = np.memmap(
                    neg_file,
                    dtype=negative_metadata["dtype"],
                    mode="r",
                    shape=tuple(negative_metadata["shape"]),
                )
                negative_activations.append(neg_memmap)

    else:
        # Cache activations for the positive and negative examples
        print("Caching activations...")

        # Cache activations for the positive and negative examples, without memmaps
        if cache_activations_save_path is None:
            positive_activations = cache_activations(
                encoder,
                positive_examples,
                batch_size,
                max_length,
                cache_dir=None,
                **kwargs,
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                cache_dir=None,
                **kwargs,
            )

        # Cache activations for the positive and negative examples, with memmaps
        else:
            positive_path = os.path.join(
                cache_activations_save_path, "positive_examples"
            )
            negative_path = os.path.join(
                cache_activations_save_path, "negative_examples"
            )
            positive_activations = cache_activations(
                encoder,
                positive_examples,
                batch_size,
                max_length,
                cache_dir=positive_path,
                **kwargs,
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                cache_dir=negative_path,
                **kwargs,
            )

    # Move model to CPU and clear GPU memory, to save VRAM for probe training
    encoder.model.to("cpu")
    torch.cuda.empty_cache()

    # Initialize probes and optimizers for each layer, and loss criterion
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, lr, device, pretrained_probes
    )

    # Train probes for all specified layers
    print("Training probes...")
    if use_parallelism:
        # Use multiprocessing to train probes in parallel
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=len(layers)) as pool:
            results = pool.starmap(
                train_layer,
                [
                    (
                        layer,
                        probes[layer],
                        optimizers[layer],
                        positive_activations[layer],
                        negative_activations[layer],
                        n_epochs,
                        batch_size,
                        n_grad_accum,
                        device,
                        cache_activations_save_path is not None,
                    )
                    for layer in layers
                ],
            )
    else:
        # Train probes sequentially
        results = [
            train_layer(
                layer,
                probes[layer],
                optimizers[layer],
                positive_activations[layer],
                negative_activations[layer],
                n_epochs,
                batch_size,
                n_grad_accum,
                device,
                cache_activations_save_path is not None,
            )
            for layer in layers
        ]

    # Print final loss for each layer and return the trained probes
    for layer, probe, losses in results:
        probes[layer] = probe
        print(f"Layer {layer} - Final Loss: {losses[-1]:.4f}")

    # Move model back to GPU and return probes
    encoder.model.to("cuda")
    return probes


def initialize_lora_adapter(model, layers, lora_params):
    # Disable gradient computation for the encoder.model
    for param in model.parameters():
        param.requires_grad = False

    # Unpack LoRA parameters
    r = lora_params.get("r", 64)
    alpha = lora_params.get("alpha", 128)
    dropout = lora_params.get("dropout", 0.0)
    bias = lora_params.get("bias", "none")
    target_modules = lora_params.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=list(range(0, max(layers) + 1)),
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapter to the model
    lora_model = get_peft_model(model, lora_config)

    return lora_model


def disable_model_gradients(lora_model):
    # Disables all model gradients
    for param in lora_model.parameters():
        param.requires_grad_(False)


def enable_model_gradients(lora_model):
    # Enables lora adapter gradients
    n_layers = lora_model.config.num_hidden_layers
    for i in range(n_layers):
        for name, param in lora_model.get_submodule("base_model.model.model.layers")[
            i
        ].named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.attacks.perturbation import PerturbationAttack
def train_oat_probe_and_model(
    llm: AutoLLM,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    lora_params={},
    probe_lr=1e-3,
    adapter_lr=5e-5,
    kl_penalty=1e-2,
    max_length=1024,
    n_steps=1000,
    n_steps_per_logging=100,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    only_choose_prompt_tokens_between=None,
    only_probe_tokens_between=None,  # New parameter for probe mask
    epsilon=10.0,
    adversary_lr=1e-3,
    pgd_iterations=32,
    clip_grad_norm=1.0,
    start_adv_training_at_step=1024,
    freeze_probes_during_adversarial_training=True,
    freeze_model_during_warmup=False,
    use_lora_adapter=True,
    checkpoint_dir=None,
    checkpoint_every=500,  # Save checkpoints every 500 steps by default
    **kwargs,
):
    print(f"Checking that n_grad_accum is 0 or a multiple of n_steps... Actual values: n_grad_accum={n_grad_accum}, n_steps={n_steps}")
    assert n_grad_accum == 0 or n_steps % n_grad_accum == 0

    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {checkpoint_dir}")

    # Initialize FLOP counters
    total_flops = 0
    probe_training_flops = 0
    adversarial_training_flops = 0
    
    wandb.init(
        project="obfuscated-activations",
        config={
            # Model metadata
            "model": {
                "name": llm.name,
                "num_layers": len(layers),
                "layers_used": layers,
            },
            # Probe metadata
            "probe": {
                "type": create_probe_fn.__name__,
                "learning_rate": probe_lr,
            },
            # LoRA metadata
            "lora": {
                "enabled": use_lora_adapter,
                "learning_rate": adapter_lr,
                "params": lora_params,
            },
            # Training metadata
            "training": {
                "batch_size": batch_size,
                "grad_accumulation_steps": n_grad_accum,
                "max_steps": n_steps,
                "kl_penalty": kl_penalty,
                "max_sequence_length": max_length,
            },
            # Dataset metadata
            "dataset": {
                "num_positive_examples": len(positive_examples),
                "num_negative_examples": len(negative_examples),
            },
            # Adversarial training params
            "adversarial": {
                "epsilon": epsilon,
                "pgd_iterations": pgd_iterations,
                "start_step": start_adv_training_at_step,
                "adversary_lr": adversary_lr,
                "freeze_probes": freeze_probes_during_adversarial_training,
                "freeze_lora_warmup": freeze_model_during_warmup,
            },
            # Masking configuration
            "masking": {
                "return_on_tokens": str(only_return_on_tokens_between),
                "choose_prompt_tokens": str(only_choose_prompt_tokens_between),
                "probe_tokens": str(only_probe_tokens_between),
            },
            # Add FLOP tracking configuration
            "flops_tracking": {
                "model_size": llm.num_params,
                "track_backward_pass": True,
            },
            # Checkpoint configuration
            "checkpointing": {
                "enabled": checkpoint_dir is not None,
                "directory": checkpoint_dir,
                "frequency": checkpoint_every,
            },
        },
        tags=[
            llm.name,
            "lora" if use_lora_adapter else "no-lora",
            "probe-only" if freeze_model_during_warmup and n_steps < start_adv_training_at_step else "probe-and-model"
        ]
    )

    # Initialize probes and optimizers for each layer
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, probe_lr, device, pretrained_probes
    )
    probes = {layer: probe.to(device) for layer, probe in probes.items()}

    # Initialize LoRA adapter
    if use_lora_adapter:
        llm._model = initialize_lora_adapter(llm._model, layers, lora_params)
        adapter_optimizer = torch.optim.AdamW(llm._model.parameters(), lr=adapter_lr)
    else:
        adapter_optimizer = None

    # Tokenize and prepare input data
    llm._tokenizer.padding_side = "right"
    positive_tokens = llm._tokenizer(
        positive_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_input_ids = positive_tokens["input_ids"]
    positive_attention_mask = positive_tokens["attention_mask"]
    negative_tokens = llm._tokenizer(
        negative_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    negative_input_ids = negative_tokens["input_ids"]
    negative_attention_mask = negative_tokens["attention_mask"]

    # Log tokenization statistics
    wandb.log({
        "tokens/pos_avg_length": torch.sum(positive_attention_mask).item() / len(positive_attention_mask),
        "tokens/neg_avg_length": torch.sum(negative_attention_mask).item() / len(negative_attention_mask),
    })

    # Target mask - where we compute the main loss
    if only_return_on_tokens_between is not None:
        zero_positive_mask = get_valid_token_mask(
            positive_input_ids, only_return_on_tokens_between
        )
        zero_negative_mask = get_valid_token_mask(
            negative_input_ids, only_return_on_tokens_between
        )
    else:
        zero_positive_mask = torch.ones_like(positive_input_ids).bool()
        zero_negative_mask = torch.ones_like(negative_input_ids).bool()

    # Probe mask - where we compute probe measurements
    if only_probe_tokens_between is not None:
        probe_positive_mask = get_valid_token_mask(
            positive_input_ids, only_probe_tokens_between
        )
        probe_negative_mask = get_valid_token_mask(
            negative_input_ids, only_probe_tokens_between
        )
    else:
        # If no probe mask specified, use the target mask
        probe_positive_mask = zero_positive_mask
        probe_negative_mask = zero_negative_mask

    # Log mask statistics
    wandb.log({
        "masks/pos_target_ratio": zero_positive_mask.float().mean().item(),
        "masks/neg_target_ratio": zero_negative_mask.float().mean().item(),
        "masks/pos_probe_ratio": probe_positive_mask.float().mean().item(),
        "masks/neg_probe_ratio": probe_negative_mask.float().mean().item(),
    })

    # This is only relevant for adversarial training
    if only_choose_prompt_tokens_between is not None:
        pos_only_choose_mask = get_valid_token_mask(
            positive_input_ids, only_choose_prompt_tokens_between
        )
        pos_only_choose_mask = pos_only_choose_mask.to(device)
        
        # Log adversarial mask statistics
        wandb.log({
            "masks/pos_prompt_ratio": pos_only_choose_mask.float().mean().item(),
        })
    else:
        pos_only_choose_mask = None

    n_examples = min(len(positive_examples), len(negative_examples))

    continue_training_next_epoch = True
    current_step = 0
    start_time = time.time()

    accumulated_toward_pgd_loss = 0
    accumulated_probe_pgd_loss = 0
    accumulated_probe_loss = 0
    accumulated_kl_loss = 0
    steps_since_last_log = 0
    info = {
        "softprompt_evals": [],
        "flops": {
            "total": 0,
            "probe_training": 0,
            "adversarial_training": 0,
        },
        "config": {
            "probe_lr": probe_lr,
            "adapter_lr": adapter_lr,
            "kl_penalty": kl_penalty,
            "max_length": max_length,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_grad_accum": n_grad_accum,
            "device": device,
            "epsilon": epsilon,
            "adversary_lr": adversary_lr,
            "pgd_iterations": pgd_iterations,
            "clip_grad_norm": clip_grad_norm,
            "start_adv_training_at_step": start_adv_training_at_step,
            "freeze_probes_during_adversarial_training": freeze_probes_during_adversarial_training,
            "freeze_model_during_warmup": freeze_model_during_warmup,
            "use_lora_adapter": use_lora_adapter,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_every": checkpoint_every,
            "layers": layers,
            "model_parameters": llm.num_params,
        }
    }
    print(f"Initializing OAT training with info: {info}")

    wrappers = []
    adversaries = []
    pgd_probe_loss = 0

    # Log initial model stats
    wandb.log({
        "model/parameters": llm.num_params,
        "training/start_time": start_time,
    })

    pbar = tqdm(total=n_steps, desc="Training LORA+Probe")

    while continue_training_next_epoch:
        # Shuffle the examples
        perm = torch.randperm(n_examples)

        for i in range(0, n_examples, batch_size):
            # Check if the batch is the last one
            if i + batch_size > n_examples:
                break

            # Get the batch
            batch_perm = perm[i : i + batch_size]
            pos_batch_input_ids = positive_input_ids[batch_perm].to(device)
            pos_batch_attention_mask = positive_attention_mask[batch_perm].to(device)
            neg_batch_input_ids = negative_input_ids[batch_perm].to(device)
            neg_batch_attention_mask = negative_attention_mask[batch_perm].to(device)
            pos_batch_zero_mask = zero_positive_mask[batch_perm].to(device).bool()
            neg_batch_zero_mask = zero_negative_mask[batch_perm].to(device).bool()
            # Get probe masks for the batch
            pos_batch_probe_mask = probe_positive_mask[batch_perm].to(device).bool()
            neg_batch_probe_mask = probe_negative_mask[batch_perm].to(device).bool()

            # Calculate tokens processed in this batch
            pos_tokens = pos_batch_input_ids.shape[0] * pos_batch_input_ids.shape[1]
            neg_tokens = neg_batch_input_ids.shape[0] * neg_batch_input_ids.shape[1]
            batch_tokens = pos_tokens + neg_tokens

            if pos_only_choose_mask is not None:
                pos_batch_only_choose_mask = (
                    pos_only_choose_mask[batch_perm].to(device).bool()
                )

            # Forward pass on positive examples
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if current_step >= start_adv_training_at_step:
                    # Print this out at the first adversarial training step
                    if current_step == start_adv_training_at_step:
                        print("FORMATTING EXAMPLES FOR ADVERSARIAL TRAINING")
                        print("This is the target mask:")
                        print(
                            repr(
                                llm._tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_zero_mask[0]]
                                )
                            )
                        )
                        print("This is the prompt mask:")
                        print(
                            repr(
                                llm._tokenizer.decode(
                                    pos_batch_input_ids[0][
                                        pos_batch_only_choose_mask[0]
                                    ]
                                )
                            )
                        )
                        print("This is the probe mask:")
                        print(
                            repr(
                                llm._tokenizer.decode(
                                    pos_batch_input_ids[0][pos_batch_probe_mask[0]]
                                )
                            )
                        )
                        
                        # Log first adversarial training step
                        wandb.log({
                            "adversarial/start_step": current_step,
                            "adversarial/start_time": time.time() - start_time,
                        })

                    # Disable model gradients
                    disable_model_gradients(llm._model)

                    # Run this eval to benchmark the effectiveness of universal soft prompts
                    # if (
                    #     current_step - start_adv_training_at_step
                    # ) % run_softprompt_eval_every == 0:
                    #     print(f"RUNNING SOFTPROMPT EVALUATION")
                    #     results = benchmark_obfuscated_softprompt(
                    #         model=lora_model,
                    #         model_layers_module="base_model.model.model.layers",
                    #         tokenizer=encoder.tokenizer,
                    #         probes=probes,
                    #         **softprompt_evals_data,
                    #     )

                    #     wandb.log({
                    #         "eval/jailbreak_rate": results["jailbreak_rate"],
                    #         "eval/avg_positive_score": results["avg_positive_score"],
                    #         "eval/avg_negative_score": results["avg_negative_score"],
                    #         "eval/success_rate_delta": results["avg_positive_score"] - results["avg_negative_score"],
                    #         "step": current_step,
                    #     })

                    #     info["softprompt_evals"].append(results)
                    #     print("Jailbreak Success Rate:", results["jailbreak_rate"])
                    #     print("Average positive score:", results["avg_positive_score"])
                    #     print("Average negative score:", results["avg_negative_score"])

                    # Train new attack from scratch against batch inputs
                    # new_attack = PerturbationAttack( # TODO

                    # )

                    # new_attack = PerturbationAttack(
                    #     num_epochs=9999,
                    #     max_steps=pgd_iterations,
                    #     epsilon=epsilon,
                    #     learning_rate=adversary_lr, # recommended: 4e-2 or 3e-2
                    #     batch_size=4,
                    #     chunk_size=4,
                    # )

                    # attack_results = new_attack.run_attack(
                    #     llm=llm,
                    #     prompts= # list of prompts in string or embedding space
                    #     harmful_responses= # list of harmful responses in string or embedding space
                    #     probe_to_obfuscate= # probe to obfuscate!
                    #     generate_final_responses= False
                    # )
                    
                    losses, wrappers = train_attack(
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=pos_batch_only_choose_mask,
                        target_mask=pos_batch_zero_mask,
                        model=llm._model,
                        tokenizer=llm._tokenizer,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=epsilon,
                        learning_rate=adversary_lr,
                        pgd_iterations=pgd_iterations,
                        probes=probes,
                        probe_mask=pos_batch_probe_mask,  # Pass probe mask
                        adversary_type="pgd",
                    )

                    pgd_toward_loss = losses["toward"]
                    pgd_probe_loss = losses["probe"]

                    # Enable model gradients on the lora adapter
                    enable_model_gradients(llm._model)

                    # Track FLOPs for adversarial training
                    pgd_tokens = pos_batch_input_ids.shape[0] * pos_batch_input_ids.shape[1] * pgd_iterations
                    pgd_flops = calculate_backward_flops(llm.num_params, pgd_tokens)
                    adversarial_training_flops += pgd_flops
                    total_flops += pgd_flops
                else:
                    pgd_toward_loss = (
                        0  # Set to 0 when adversarial training is not used
                    )
                    pgd_probe_loss = 0
                    wrappers = []

                for wrapper in wrappers:
                    wrapper.enabled = True

                pos_output = llm._model(
                    input_ids=pos_batch_input_ids,
                    output_hidden_states=True,
                )
                if current_step == 0:
                    print(f"Number of layers in model: {len(pos_output.hidden_states)}")
                    print(f"Getting layer indices from hidden states (plus one): {layers}")
                pos_acts = {
                    layer: pos_output.hidden_states[layer + 1] for layer in layers
                }
                
                # Track forward pass FLOPs
                forward_flops = calculate_forward_flops(llm.num_params, pos_tokens)
                total_flops += forward_flops
                if current_step < start_adv_training_at_step:
                    probe_training_flops += forward_flops
                else:
                    adversarial_training_flops += forward_flops

            # Compute the positive probe losses using probe mask
            pos_loss = 0
            layer_losses = {}
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    pos_targets = torch.ones_like(
                        pos_acts[layer][..., 0], device=device
                    )
                    pos_layer_loss = probe.compute_loss(
                        pos_acts[layer],
                        pos_targets,
                        mask=pos_batch_probe_mask,  # Use probe mask
                    )
                    pos_loss += pos_layer_loss
                    layer_losses[f"layer_{layer}_pos_loss"] = pos_layer_loss.item()

            # Track backward pass FLOPs for positive examples
            backward_flops = calculate_backward_flops(llm.num_params, pos_tokens)
            total_flops += backward_flops
            if current_step < start_adv_training_at_step:
                probe_training_flops += backward_flops
            else:
                adversarial_training_flops += backward_flops

            # Backward pass on positive examples
            pos_loss.backward(retain_graph=True)

            for wrapper in wrappers:
                wrapper.enabled = False

            # Forward pass on negative examples
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                neg_output = llm._model(
                    input_ids=neg_batch_input_ids,
                    output_hidden_states=True,
                )
                neg_logits = neg_output.logits
                neg_acts = {
                    layer: neg_output.hidden_states[layer + 1] for layer in layers
                }
                
                # Track forward pass FLOPs for negative examples
                forward_flops = calculate_forward_flops(llm.num_params, neg_tokens)
                total_flops += forward_flops
                if current_step < start_adv_training_at_step:
                    probe_training_flops += forward_flops
                else:
                    adversarial_training_flops += forward_flops

            # Compute the negative probe losses using probe mask
            neg_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    neg_targets = torch.zeros_like(
                        neg_acts[layer][..., 0], device=device
                    )
                    neg_layer_loss = probe.compute_loss(
                        neg_acts[layer],
                        neg_targets,
                        mask=neg_batch_probe_mask,  # Use probe mask
                    )
                    neg_loss += neg_layer_loss
                    layer_losses[f"layer_{layer}_neg_loss"] = neg_layer_loss.item()

            # Backward pass on negative examples
            neg_loss.backward(retain_graph=True)

            # Track backward pass FLOPs for negative examples
            backward_flops = calculate_backward_flops(llm.num_params, neg_tokens)
            total_flops += backward_flops
            if current_step < start_adv_training_at_step:
                probe_training_flops += backward_flops
            else:
                adversarial_training_flops += backward_flops

            # Get model loss only if we're in a phase in which the model is being trained...
            if not freeze_model_during_warmup or not (
                    current_step < start_adv_training_at_step
                ):
                # Compute KL divergence of logits from base model logits
                with torch.no_grad():
                    llm._model.disable_adapter_layers()
                    base_neg_output = llm._model(
                        input_ids=neg_batch_input_ids,
                        # attention_mask=neg_batch_attention_mask,
                    )
                    llm._model.enable_adapter_layers()

                # Get logits only for masked positions
                base_logits = base_neg_output.logits[neg_batch_zero_mask]
                model_logits = neg_logits[neg_batch_zero_mask]

                kl_loss = F.kl_div(
                    F.log_softmax(base_logits, dim=-1),
                    F.softmax(model_logits, dim=-1),
                    reduction="batchmean",
                )

                # Track FLOPs for KL divergence forwarding & backward
                kl_flops = calculate_forward_flops(llm.num_params, neg_tokens) + calculate_backward_flops(llm.num_params, neg_tokens)
                total_flops += kl_flops
                if current_step < start_adv_training_at_step:
                    probe_training_flops += kl_flops
                else:
                    adversarial_training_flops += kl_flops

                # Backward pass on KL divergence
                (kl_loss / (kl_loss.detach() + 1e-8) * kl_penalty).backward()
                accumulated_kl_loss += kl_loss.item()

            # Accumulate losses
            accumulated_probe_loss += pos_loss.item() + neg_loss.item()
            accumulated_toward_pgd_loss += (
                pgd_toward_loss
            )
            accumulated_probe_pgd_loss += pgd_probe_loss
            steps_since_last_log += 1

            # Perform optimization step after accumulating gradients
            if (i // batch_size + 1) % n_grad_accum == 0 or (
                i + batch_size
            ) >= n_examples:

                # Clip the gradients if specified
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        llm._model.parameters(), clip_grad_norm
                    )
                    all_probe_params = [
                        param
                        for probe in probes.values()
                        for param in probe.parameters()
                    ]
                    torch.nn.utils.clip_grad_norm_(all_probe_params, clip_grad_norm)

                # Optimize probes only when not using adversarial training
                if not freeze_probes_during_adversarial_training or not (
                    # In adversarial phase
                    current_step > start_adv_training_at_step
                ):
                    for optimizer in optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad()

                # Optimize the adapter
                if not freeze_model_during_warmup or not (
                    # In warmup phase
                    current_step < start_adv_training_at_step
                ):
                    if adapter_optimizer is not None:
                        adapter_optimizer.step()
                        adapter_optimizer.zero_grad()

            current_step += 1

            # Save checkpoints if specified
            if checkpoint_dir and current_step % checkpoint_every == 0:
                # Save probes
                probe_path = os.path.join(checkpoint_dir, f"probes_step_{current_step}.pt")
                save_probes(probes=probes, save_path=probe_path)
                print(f"Saved probe checkpoint to {probe_path}")
                
                # Save LoRA model if using it
                if use_lora_adapter:
                    model_path = os.path.join(checkpoint_dir, f"lora_model_step_{current_step}")
                    llm._model.save_pretrained(model_path)
                    print(f"Saved model checkpoint to {model_path}")
                
                # Save training info
                info_path = os.path.join(checkpoint_dir, f"training_info_step_{current_step}.json")
                current_info = {
                    "step": current_step,
                    "probe_loss": accumulated_probe_loss / max(steps_since_last_log, 1),
                    "kl_loss": accumulated_kl_loss / max(steps_since_last_log, 1),
                    "flops": {
                        "total": total_flops,
                        "probe_training": probe_training_flops,
                        "adversarial_training": adversarial_training_flops,
                    },
                    "timestamp": time.time(),
                    "elapsed_time": time.time() - start_time
                }
                
                current_info.update({
                    "pgd_toward_loss": accumulated_toward_pgd_loss / max(steps_since_last_log, 1),
                    "pgd_probe_loss": accumulated_probe_pgd_loss / max(steps_since_last_log, 1),
                })
                
                with open(info_path, "w") as f:
                    json.dump(current_info, f)
                print(f"Saved training info to {info_path}")
                
                # Log checkpoint in wandb
                wandb.log({
                    "checkpoint/step": current_step,
                    "checkpoint/path": os.path.abspath(checkpoint_dir),
                    "checkpoint/elapsed_time": time.time() - start_time,
                })

            if current_step % n_steps_per_logging == 0:
                avg_probe_loss = accumulated_probe_loss / steps_since_last_log
                avg_kl_loss = accumulated_kl_loss / steps_since_last_log
                avg_toward_pgd_loss = (
                    accumulated_toward_pgd_loss / steps_since_last_log
                )
                avg_probe_pgd_loss = (
                    accumulated_probe_pgd_loss / steps_since_last_log
                )
                avg_total_loss = avg_probe_loss + avg_kl_loss

                # Prepare metrics to log
                metrics = {
                    "train/total_loss": avg_total_loss,
                    "train/probe_loss": avg_probe_loss,
                    "train/kl_loss": avg_kl_loss,
                    "step": current_step,
                    "elapsed_time": time.time() - start_time,
                    "tokens_processed": current_step * batch_size * max_length,
                }
                
                # Add per-layer losses
                for key, value in layer_losses.items():
                    metrics[f"train/{key}"] = value

                metrics.update({
                    "train/pgd_toward_loss": avg_toward_pgd_loss,
                    "train/pgd_probe_loss": avg_probe_pgd_loss,
                })

                # Log FLOP metrics
                metrics.update({
                    "flops/total": total_flops,
                    "flops/probe_training": probe_training_flops,
                    "flops/adversarial_training": adversarial_training_flops,
                })
                
                # Log GPU memory usage
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics[f"system/gpu{i}_memory_used"] = torch.cuda.memory_allocated(i) / (1024**3)  # in GB
                
                # Update info dictionary
                info["flops"]["total"] = total_flops
                info["flops"]["probe_training"] = probe_training_flops
                info["flops"]["adversarial_training"] = adversarial_training_flops

                # Log all metrics to wandb
                wandb.log(metrics)

                log_message = (
                    f"Step: {current_step}/{n_steps}, "
                    f"Time: {convert_seconds_to_time_str(time.time() - start_time)}, "
                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                    f"Avg Probe Loss: {avg_probe_loss:.4f}, "
                    f"Avg KL Loss: {avg_kl_loss:.4f}"
                )

                log_message += f", Avg Toward PGD Loss: {avg_toward_pgd_loss:.4f}"
                log_message += f", Avg Probe PGD Loss: {avg_probe_pgd_loss:.4f}"

                print(log_message)

                # Reset accumulators
                accumulated_toward_pgd_loss = 0
                accumulated_probe_pgd_loss = 0
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0
                steps_since_last_log = 0

            if current_step >= n_steps:
                continue_training_next_epoch = False
                break

            pbar.update(1)  # Update progress bar

    # Log final summary stats
    wandb.run.summary.update({
        "final_probe_loss": avg_probe_loss,
        "final_kl_loss": avg_kl_loss,
        "final_total_loss": avg_total_loss,
        "training_duration": time.time() - start_time,
        "total_steps": current_step,
        "total_flops": total_flops,
        "probe_training_flops": probe_training_flops,
        "adversarial_training_flops": adversarial_training_flops,
    })

    wandb.run.summary.update({
        "final_pgd_toward_loss": avg_toward_pgd_loss,
        "final_pgd_probe_loss": avg_probe_pgd_loss,
    })

    # Close wandb run
    wandb.finish()
    return probes, llm._model, info


def save_probes(probes, save_path):
    # Save a list of probes to a file
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(probes, save_path)


def load_probes(load_path):
    # Load a list of probes from a file
    return torch.load(load_path)
