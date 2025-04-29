# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
- Use `uv` for package management: `uv pip install -r requirements.txt`
- Activate virtual environment: `source .venv/bin/activate`
- Set up API keys: `export OPENAI_API_KEY="your-key-here"` (for evaluations)

## Build/Test Commands
- Train backdoor models: `python train_backdoored_model_no_explicit_obfuscation.py --N_STEPS=200000 --MODEL_LR=2e-06`
- Evaluate defenses: `python evaluate_defenses.py`
- Run OAT training: `python compute_oat_probes.py --masking-type=generation --probe-type=linear`

## Code Style Guidelines
- Imports: Standard library first, third-party libraries next, local imports last
- Type hints: Use typing module (Optional, Dict, List) for all function signatures
- Variable naming: snake_case for variables/functions, CamelCase for classes, UPPER_SNAKE_CASE for constants
- Docstrings: Google-style with Args, Returns sections for functions and classes
- Error handling: Use assertions with descriptive messages, raise specific exceptions
- PyTorch patterns: Use @torch.inference_mode() or context managers for gradient control
- Module structure: Group related functionality, maintain separation of concerns
- Functions: Single responsibility, clear parameter naming, reasonable defaults