import os
import spacy
from typing import List

# Load the English model
_nlp = spacy.load("en_core_web_sm")

# Paragraph segmentation: Splitting up each sentence of the paragraph
# It will be then chunked for alternate phrasing for every sentence.
def segment_para(para:str) -> List[str]:
    # Process the paragraph
    doc = _nlp(para)
    
    # Extract the sentences
    sentences = [sent.text for sent in doc.sents]
    
    return sentences