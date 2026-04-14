from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.4-nano"


def generate_text(prompt: str) -> str:
    try:
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=300
        )
        return response.output[0].content[0].text.strip()
    except Exception:
        return "LLM generation failed."
