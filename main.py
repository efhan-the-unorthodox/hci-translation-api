from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from translator_pipe.p1_translator import *
from translator_pipe.p2_chunking import *
from translator_pipe.p4_alt_builder import *
from translator_pipe.p1_5_sentence_segmt import *
from fastapi.middleware.cors import CORSMiddleware
import ast

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allow only your React app origins
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"],         # allow all headers
)


@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


"""
init_translation 
This post request receives the raw source sentence from the frontend and will send it through the first two stages of 
the translation pipeline. It will `translate` to English. Then with the bpe tokens from the OpenAI ChatCompletion, 
it will then pass through spacy and do the necessary chunking process. See it for yourself in the pipeline.
"""
class TranslationInput(BaseModel):
    input_sent:str
    source_lang: str
    
@app.post("/translate")
def init_translation(transInput: TranslationInput):
    tokens, logrows = translate(transInput.input_sent, transl_frm=transInput.source_lang)
    chunks, plaintext = chunk_sentence(tokens)
    return {"chunks": chunks, "plaintext": plaintext}

class ParaInput(BaseModel):
    input_para: str
    source_lang :str

@app.post("/translate_para")
def para_translation(paraInput:ParaInput):
    tsl_para = translate_para(paraInput.input_para, transl_frm=paraInput.source_lang)
    sentences = segment_para(tsl_para)
    
    return sentences
    
class SentInput(BaseModel):
    input_sent:str 
@app.post("/sentence_chunking")
def generate_sent_chunks(sentInput:SentInput):
    chunks = chunk_sentence_2(sentInput.input_sent)
    return chunks
    
class AlternativeInput(BaseModel):
    chunk_text: str
    whole_sent:str
    
@app.post("/alternate_chunking")
def generate_alternatives(data: AlternativeInput):
    alt_resp = generate_alternate_phrasing(data.chunk_text, data.whole_sent)
    if isinstance(alt_resp, str):
        parsed_list = ast.literal_eval(alt_resp)
    return parsed_list