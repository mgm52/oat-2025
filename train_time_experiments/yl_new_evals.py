import torch
import os

defender_checkpoint_dir = "defender_checkpoint_dir=/workspace/oat-2025/train_time_experiments/oat-gemma-apr2-checks/gemma2_lora_oat_generation_linear"

# attacker_checkpoint_dir = "/workspace/storage/checkpoints/attack/pgd/2025-04-14_1633"
# attacker_checkpoint = "checkpoint_step_33.pt"

attacker_checkpoint_dir = "/workspace/storage/checkpoints/attack/pgd/2025-04-14_1519"
attacker_checkpoint = "checkpoint_step_0.pt"


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

for adversary in adversaries:
    for param in adversary.parameters():
        param.requires_grad = False

# Get adversary weight

# Print model
lora_model


# Check current output on trained sample
bad_example = split_jailbreaks_dataset.examples_bad.train[0]
bad_prompt, bad_generation = bad_example.split('model\n')
bad_prompt += "model\n"
encoder.tokenizer.padding_side = "left"
bad_prompt_tokens = encoder.tokenizer(bad_prompt, return_tensors="pt"    ,
                                      padding=True,
                                      max_length=1024,
                                      add_special_tokens=False,  # special tokens already in dataset!
).input_ids
out = lora_model.generate(bad_prompt_tokens.to("cuda"),
                          max_new_tokens=50,
                          return_dict_in_generate=True, output_scores=True)
# Decode the generated tokens to get the model's output
generated_text = encoder.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

# Check current output on val sample
bad_example = split_jailbreaks_dataset.examples_bad.val[0]
bad_prompt, bad_generation = bad_example.split('model\n')
bad_prompt += "model\n"
encoder.tokenizer.padding_side = "left"
bad_prompt_tokens = encoder.tokenizer(bad_prompt, return_tensors="pt"    ,
                                      padding=True,
                                      max_length=1024,
                                      add_special_tokens=False,  # special tokens already in dataset!
).input_ids
out = lora_model.generate(bad_prompt_tokens.to("cuda"),
                          max_new_tokens=50,
                          return_dict_in_generate=True, output_scores=True)
# Decode the generated tokens to get the model's output
generated_text = encoder.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")


# Add forward hook to model
# How exactly is the mask applied? Which part can have the mask?
# Note: The bfloat16 conversion is necessary to avoid type errors
def add_embedding_perturb(model, input, output):
    with torch.no_grad():
        # return output + list(adversaries[0].parameters())[0][:, :PROMPT_SEQ_LEN].to(torch.bfloat16)
        print(output.shape)
        print(PERTURBATION.shape)
        padded_perturbation = torch.nn.functional.pad(PERTURBATION, (0, 0, 0, output.shape[1] - PERTURBATION.shape[1]))
        print(padded_perturbation.shape)
        return output + padded_perturbation
print("Adding forward hook")
hook = lora_model.base_model.model.model.embed_tokens.register_forward_hook(add_embedding_perturb)


# Check current output on trained sample
bad_example = split_jailbreaks_dataset.examples_bad.train[0]
bad_prompt, bad_generation = bad_example.split('model\n')
bad_prompt += "model\n"
encoder.tokenizer.padding_side = "left"
bad_prompt_tokens = encoder.tokenizer(bad_prompt, return_tensors="pt"    ,
                                      padding=True,
                                      max_length=1024,
                                      add_special_tokens=False,  # special tokens already in dataset!
).input_ids
PROMPT_SEQ_LEN = bad_prompt_tokens.shape[1]
PERTURBATION = next(adversaries[0].parameters())[:1, :PROMPT_SEQ_LEN].to(torch.bfloat16)
out = lora_model.generate(bad_prompt_tokens.to("cuda"),
                          max_new_tokens=50,
                          return_dict_in_generate=True, output_scores=True)
# Decode the generated tokens to get the model's output
generated_text = encoder.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")


# Check current output on val sample

# Repeat the above for 9b model