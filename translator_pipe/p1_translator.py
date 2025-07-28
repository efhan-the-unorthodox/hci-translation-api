import os, dotenv, json, math
from openai import OpenAI
from openai.types.chat import ChatCompletion

dotenv.load_dotenv()
client = OpenAI()
secret_key = os.getenv("OPENAI_API_KEY")


def translation_init(input: str, model="gpt-4o", source_lang="Chinese(Simplified)"):
    main_instruction = f"You are a translator and your objective is to translate from {source_lang} to English (United Kingdom). The response should only contain the translated text."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": main_instruction},
            {"role": "user", "content": f"Please Translate:{input}"},
        ],
        logprobs=True,
        top_logprobs=4,
    )

    return response


def translation_processing(response: ChatCompletion):
    logprobs_content = response.choices[0].logprobs.content  # rows
    logprobs_tokens = [row.token for row in logprobs_content]  # tokens

    return logprobs_tokens, logprobs_content  # caller can feed tokens to chunker


def translate(input: str, model="gpt-4o", transl_frm="Chinese(Simplified)"):

    resp = translation_init(input, model, transl_frm)

    tokens, rows = translation_processing(resp)
    
    return tokens, rows
