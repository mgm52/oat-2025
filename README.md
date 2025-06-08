# Obfuscated Adversarial Training

This repo is for training and evaluation of the **OAT jailbreak defense (Obfuscated Adversarial Training)** proposed by [Bailey et al](https://arxiv.org/pdf/2412.09565). OAT is a post-training routine to improve the adversarial robustness of latent-space probes.

The broader aim of this repo is to provide an evaluation framework for comparing _LLM jailbreaks_ vs _finetuning defenses_ vs _latent-space probes_, against FLOP cost a la [Boreiko et al](https://arxiv.org/pdf/2410.16222v1) and StrongREJECT score per [Souley et al](https://arxiv.org/abs/2402.10260).

Repo overview:
- `oat_evaluation/` contains the evaluation framework.
- `oat_training/` adapts Bailey et al's original OAT training code.

Run `installation.sh` to get started. Then `oat_evaluation/scripts/eval_scripts/full_eval_test_multi_gpu.py` to kick off evals.

---

### Early results

More compute-expensive OAT runs (more OAT steps or more mid-training attack steps) produce predictably lower StrongREJECT scores against soft-suffix attacks.

> _**Details**: The graph below aggregates 1077 tests. Each test involves learning a universal soft-suffix against 1000 harmful samples, then evaluating the jailbreak against 100 more, extracting StrongREJECT (SR) score on outputs. Attacker FLOPs vary with suffix token length and number of attacker training steps._

<img src="https://github.com/user-attachments/assets/0c969a9f-cfae-4485-b381-a2282fd6dded" width="500">

However, results so far against hard-token attacks are disappointing, with OAT performing no better against 20 PAIR attacks than the base model, despite Circuit Breakers overcoming them easily:


![image](https://github.com/user-attachments/assets/301dfef4-8393-4136-b200-ed2045b503dc)


---

### Status

Currently implemented attacks:
- Soft-suffixes
- Soft-perturbations
- PAIR
- FLRT
- GCG

Currently acquired defenses:
- OAT
- LAT ([checkpoints](https://huggingface.co/LLM-LAT/robust-llama3-8b-instruct))
- Circuit Breakers ([checkpoints](https://huggingface.co/collections/GraySwanAI/model-with-circuit-breakers-668ca12763d1bc005b8b2ac3))
- Refat ([implementation](https://github.com/mgm52/refat-unofficial))
