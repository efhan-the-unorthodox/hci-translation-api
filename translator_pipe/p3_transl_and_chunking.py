import os, dotenv, json, math
# OpenAI-related translation functionality has been disabled for now.
# Removed: from openai import OpenAI
# Removed: from openai.types.chat import ChatCompletion

import re
from typing import List, Tuple, Dict
import spacy
from spacy.matcher import Matcher

from p1_translator import *
from p2_chunking import *

dotenv.load_dotenv()

def user_input():
    print("Enter a complete sentence to translate")
    return input()


def translate():

    return

def main():
    translate()


if __name__ == "__main__":
    main()
