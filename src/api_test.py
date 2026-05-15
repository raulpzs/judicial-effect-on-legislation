from openai import OpenAI
from dotenv import load_dotenv
import os

#load_dotenv()

#print("Key loaded:", bool(os.getenv("LI_AZURE_OPENAI_KEY")))

from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]  # project root if script is in src/file
ENV_PATH = BASE_DIR / ".env"

print("ENV path:", ENV_PATH)
print("ENV exists:", ENV_PATH.exists())

load_dotenv(dotenv_path=ENV_PATH, override=True)

key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", bool(key))
print("Key prefix:", key[:12] if key else None)
print("Key suffix:", key[-6:] if key else None)

client = OpenAI(api_key=key, base_url="https://dtapi.openai.azure.com/openai/v1/")


response = client.responses.create(
    model="gpt-5",
    input="Say hello in one short sentence."
)

print("RAW RESPONSE:")
print(response)

print("\nOUTPUT TEXT:")
print(response.output_text)