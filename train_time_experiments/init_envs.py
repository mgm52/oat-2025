import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
