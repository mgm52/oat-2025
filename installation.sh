# git clone https://github.com/abhay-sheshadri/sae_experiments
# cd sae_experiments
# git checkout jordanexps
pip cache purge

pip install torch
pip install transformers
pip install --upgrade huggingface_hub
pip install datasets
pip install git+https://github.com/TransformerLensOrg/TransformerLens
pip install circuitsvis
pip install peft
pip install simple_parsing
pip install natsort
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install wandb
pip install numpy==1.26.4
pip install git+https://github.com/dsbowen/strong_reject.git@main
pip install git+https://github.com/serteal/nanoflrt.git@main  # FLRT attack

# sudo apt-get install python-tk python3-tk tk-dev
# huggingface-cli login
# wandb login