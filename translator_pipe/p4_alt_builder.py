# # backend/alt_builder.py
# """
# build_alternatives_for_chunk(logprob_rows, span_tokens, ...)
# returns a list of dicts: {"text": alt_phrase, "score": logprob_sum}
# """
# import os, dotenv, pathlib, json
# from itertools import combinations, product
# from typing import List, Dict, Tuple
# from openai import OpenAI

# dotenv.load_dotenv()
# json_sk = json.loads(os.getenv("OPENAI_API_KEY"))
# secret_key = json_sk["OPENAI_API_KEY"]
# client = OpenAI(api_key=secret_key)

# _dir = pathlib.Path(os.getcwd()).resolve()


# def _surface(tok: str) -> str:
#     """Remove leading 'Ġ' (space marker) but NOT interior spaces."""
#     return tok[1:] if tok.startswith("Ġ") else tok


# def build_alternatives_for_chunk(
#     logprob_rows: List,  # response.choices[0].logprobs.content
#     span_tokens: Tuple[int, int],  # (start_idx, end_idx) inclusive
#     max_replacements: int = 2,  # ≤ how many tokens we let change
#     keep_top_n: int = 5,  # return this many alts
#     score_margin: float = 3.0,  # drop alts worse than orig+margin
# ) -> List[Dict]:
#     s, e = span_tokens
#     rows = logprob_rows[s : e + 1]
#     orig_tokens = [_surface(r.token) for r in rows]
#     orig_score = sum(r.logprob for r in rows)

#     # original phrase always included
#     phrases: Dict[str, float] = {" ".join(orig_tokens): orig_score}

#     # helper: choose alt tokens for k positions
#     L = len(rows)
#     for k in range(1, max_replacements + 1):
#         for positions in combinations(range(L), k):
#             alt_lists = []
#             for pos in positions:
#                 top = sorted(
#                     rows[pos].top_logprobs.items(), key=lambda kv: kv[1], reverse=True
#                 )[:3]
#                 # top == list of (alt_token, alt_logprob)
#                 alt_lists.append(top)
#             # Cartesian product over chosen positions
#             for repl in product(*alt_lists):
#                 new_tokens = orig_tokens[:]
#                 new_score = orig_score
#                 for pos, (alt_tok, alt_lp) in zip(positions, repl):
#                     new_tokens[pos] = _surface(alt_tok)
#                     new_score += alt_lp - rows[pos].logprob
#                 phrase = " ".join(new_tokens)
#                 if new_score >= orig_score - score_margin:
#                     phrases[phrase] = new_score

#     # sort by score, keep_top_n
#     best = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)[:keep_top_n]
#     return [{"text": p, "score": s} for p, s in best]


# def whack_alt_phrasing(chunk: str, sug_count: int = 3):
#     model = "gpt-4o-mini"
#     with open(_dir / "translator_pipe" / "phrasingprompt.txt", "r") as f:
#         instructions = f.read()
#     response = client.responses.create(
#         model=model,
#         instructions=instructions,
#         input=[
#             {
#                 "role": "user",
#                 "content": f"Give me no more than {sug_count} alternatives to saying: {chunk}",
#             },
#         ],
#     )

#     return response


# def alt_builder(chunk: str) -> list[str]:
#     resp = whack_alt_phrasing(chunk)
#     list_alts: list[str] = resp.output_text
#     return list_alts


# def generate_alternate_phrasing(chunk:str, whole_sent:str, sug_count:int = 3):
#     model = "gpt-4o-mini"
#     with open(_dir / "translator_pipe" / "rephrase_prompt.txt", "r") as f:
#         instructions = f.read()
#     response = client.responses.create(
#         model=model,
#         instructions=instructions,
#         input=[
#             {
#                 "role": "user",
#                 "content": f"Original sentence: {whole_sent}\nSuggest up to {sug_count} (where necessary) alternative phrasings for: '{chunk}'",
#             },
#         ],
#     )
#     return response.output_text


# def batch_whack_alt_phrasing(chunks: list[str], sug_count: int = 3):
#     model = "gpt-4o-mini"
#     from pathlib import Path
#     with open(_dir / "testing_and_research" / "phrasingprompt.txt", "r") as f:
#         instructions = f.read()
#     # Build a single prompt for all chunks
#     numbered_chunks = "\n".join([f"{i+1}. \"{chunk}\"" for i, chunk in enumerate(chunks)])
#     prompt = (
#         f"Give me at least {sug_count} alternatives to saying each of the following phrases. "
#         "Return the result as a Python dictionary mapping each phrase to a list of alternatives.\n\n"
#         f"{numbered_chunks}"
#     )
#     response = client.responses.create(
#         model=model,
#         instructions=instructions,
#         input=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#     )
#     return response


# def batch_alt_builder(chunks: list[str], sug_count: int = 3) -> dict:
#     resp = batch_whack_alt_phrasing(chunks, sug_count)
#     # Expecting output_text to be a Python dict as a string
#     import ast
#     try:
#         alt_dict = ast.literal_eval(resp.output_text)
#         # Ensure all originals are included as first option
#         for chunk in chunks:
#             if chunk not in alt_dict:
#                 alt_dict[chunk] = [chunk]
#             elif chunk not in alt_dict[chunk]:
#                 alt_dict[chunk] = [chunk] + alt_dict[chunk]
#         return alt_dict
#     except Exception as e:
#         # Fallback: return each chunk mapped to itself
#         return {chunk: [chunk] for chunk in chunks}


# # if __name__ == "__main__":
# #     alt_builder("A study spanning 80 years")