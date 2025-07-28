import os, dotenv, json, math
from openai import OpenAI
from openai.types.chat import ChatCompletion

import re
from typing import List, Tuple, Dict
import spacy
from spacy.matcher import Matcher

from p1_translator import *
from p2_chunking import *

dotenv.load_dotenv()
client = OpenAI()
secret_key = os.getenv("OPENAI_API_KEY")


def user_input():
    print("Enter a complete sentence to translate")
    return input()


def transl_and_chunk():
    source_text = user_input()
    tokens, log_rows = translate(source_text)
    
    chunks, plain_sent = chunk_sentence(tokens)
    
    # Translated text
    print("Translated Sentence: ", plain_sent)
    
    # Chunked Mappings
    print("Chunks:")
    for chunk in chunks:
        print(chunk['id'], chunk['label'], chunk['text'], chunk['tok_range'], "\n")
    
    return

def main():
    transl_and_chunk()


if __name__ == "__main__":
    main()
