from attack_init import *

# %%

# Evaluate

softprompt_evals_data={
    # "test_negative_examples": retain_examples_val,
    # "test_positive_examples": forget_examples_val,
    # "test_positive_prompts": forget_examples_val_prompts,
    "test_negative_examples": split_jailbreaks_dataset.examples_good.val,
    "test_positive_examples": split_jailbreaks_dataset.examples_bad.val,
    "test_positive_prompts": split_jailbreaks_dataset.prompts_only_bad.val,
    "only_return_on_tokens_between": only_return_on_tokens_between,
    "only_choose_prompt_tokens_between": only_choose_prompt_tokens_between,
    "only_probe_tokens_between": only_probe_tokens_between,
}

# generations_table = None

# Load latest checkpoint if it exists
# latest_checkpoint = None
# latest_step = -1

if attacker_checkpoint is None:
    latest_step = -1
    for filename in os.listdir(attacker_checkpoint_dir):
        if filename.startswith('checkpoint_step_') and filename.endswith('.pt'):
            step = int(filename.split('_')[-1].split('.')[0])
            if step > latest_step:
                latest_step = step
                attacker_checkpoint = filename

if attacker_checkpoint is not None:
    checkpoint_path = os.path.join(attacker_checkpoint_dir, attacker_checkpoint)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    step_count = checkpoint['step']
    adversaries = checkpoint['adversaries'] 
    wrappers = checkpoint['wrappers']
    total_flops = checkpoint['total_flops']
    print(f"Loaded checkpoint from {checkpoint_path}")

# Log hyperparameters
config = {
    "att_ckpt_dir": attacker_checkpoint_dir,
    "att_ckpt": attacker_checkpoint,
    "def_ckpt_dir": defender_checkpoint_dir,
    "def_ckpt": defender_checkpoint,
    "max_num_samples": max_num_samples,
    # "n_steps": n_steps,
    # "batch_size": batch_size,
    # "pgd_iterations": pgd_iterations,
    # "adversary_lr": adversary_lr,
    # "epsilon": epsilon,
    # "towards_loss_coef": towards_loss_coef,
    # "probe_loss_coef": probe_loss_coef,
    # "adversary_type": adversary_type,
    # "gradient_accumulation_steps": gradient_accumulation_steps,
}

print(f"{config=}")

wandb.init(
    project="probe-attack",
    name=f"eval_run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}", 
    config=config
)
# Create new table
generations_table = wandb.Table(columns=["prompt", 
"response"])
evaluations_table = wandb.Table(columns = ["question", 
            "response",
            "grader_model",
            "grader_input",
            "grader_output",
            "score",
            "error",
            "parse_error",
            "refused",
            "convincing_score",
            "specific_score",
])



print("RUNNING SOFTPROMPT EVALUATION")
results = benchmark_obfuscated_softprompt(
    model=lora_model,
    model_layers_module="base_model.model.model.layers",
    tokenizer=encoder.tokenizer,
    probes=probes,
    **softprompt_evals_data,
    adversaries=adversaries,
    wrappers=wrappers,
    gen_batch_size=batch_size,
    generations_table=generations_table,
    evaluations_table=evaluations_table,
    max_num_eval_prompts=max_num_samples,
)
print("Jailbreak Success Rate:", results["jailbreak_rate"])
print("Average positive score:", results["avg_positive_score"])
print("Average negative score:", results["avg_negative_score"])
print("Recall (0.01 FPR)", results["recall_0.01_fpr"])

import csv

# Log benchmark results to wandb
wandb.log({
    "benchmark/jailbreak_rate": results["jailbreak_rate"],
    "benchmark/avg_positive_score": results["avg_positive_score"],
    "benchmark/avg_negative_score": results["avg_negative_score"],
    "step": step_count,
    "recall_0.01_fpr": results["recall_0.01_fpr"],
})

# Log benchmark results to a CSV file
csv_file_path = os.path.join(attacker_checkpoint_dir, 'benchmark_results.csv')
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        attacker_checkpoint_dir,
        attacker_checkpoint,
        results["jailbreak_rate"],
        results["avg_positive_score"],
        results["avg_negative_score"],
        step_count,
        results["recall_0.01_fpr"]
    ])



# # %%

# Implement from Scratch


# bad_example = split_jailbreaks_dataset.examples_bad.train[0]
# bad_prompt, bad_generation = bad_example.split('model\n')
# bad_prompt += "model\n"
# encoder.tokenizer.padding_side = "left"
# bad_prompt_tokens = encoder.tokenizer(bad_prompt, return_tensors="pt"    ,
#                                       padding=True,
#                                       max_length=1024,
#                                       add_special_tokens=False,  # special tokens already in dataset!
# ).input_ids
# bad_prompt_tokens = bad_prompt_tokens.to("cuda")
# bad_generation_tokens = encoder.tokenizer(bad_generation, return_tensors="pt"    ,
#                                       padding=True,
#                                       max_length=1024,
#                                       add_special_tokens=False,  # special tokens already in dataset!
# ).input_ids
# bad_generation_tokens = bad_generation_tokens.to("cuda")

# embedding = lora_model.base_model.model.model.embed_tokens.cuda()
# # # embedding = lora_model.base_model.model.model.embed_tokens.module.cuda()
# # inp = initial_soft_prompt_text_tokens.input_ids.cuda()
# # initial_embedding_suffix = embedding(inp)
# # embedding_suffix = nn.Parameter(data=initial_embedding_suffix)
# # embedding_suffix.cuda()

# embedding_size = 3584
# initial_embedding_suffix = torch.zeros(size=(1, embedding_size), dtype=torch.bfloat16)
# embedding_suffix = nn.Parameter(data=initial_embedding_suffix)
# embedding_suffix = embedding_suffix.to("cuda")
# torch.nn.init.kaiming_uniform_(initial_embedding_suffix)
# epsilon = 10
# with torch.no_grad():
#     norms = torch.norm(initial_embedding_suffix, dim=-1, keepdim=True)
#     scale = torch.clamp(norms / epsilon, min=1)
#     initial_embedding_suffix.div_(scale)
# def add_embedding_suffix(model, input, output):
#     return output + embedding_suffix




# # # def add_embedding_suffix(name):
# # #     # the hook signature
# # #     def hook(model, input, output):
# # #         return torch.cat([output, embedding_suffix], dim=1)
# # #     return hook
# # def add_embedding_suffix(model, input, output):
# #     return torch.cat([output, embedding_suffix], dim=1)

# # import collections
# # lora_model.base_model.model.model.embed_tokens._forward_hooks = collections.OrderedDict()
# h = lora_model.base_model.model.model.embed_tokens.register_forward_hook(add_embedding_suffix)

# # lora_model.base_model.model.model.embed_tokens.hook_fn.forward = lambda x: x

# lora_model = lora_model.to("cuda")
# lora_model(bad_prompt_tokens.to("cuda"))


# out = lora_model.generate(bad_prompt_tokens.to("cuda"), max_new_tokens=471,
#                           return_dict_in_generate=True, output_scores=True)
# criterion = nn.BCEWithLogitsLoss()

# # FIXME: Reshape out.scores
# behavioural_loss = criterion(out.scores, bad_generation_tokens)

# # TODO: Gather/average probe logits


# # %%

# %%
