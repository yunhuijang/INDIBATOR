"""Configuration module that automatically loads .env file."""
from dotenv import load_dotenv

# Load environment variables from .env file once when this module is imported
load_dotenv()

