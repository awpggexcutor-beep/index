# -*- coding: utf-8 -*-
# @Author: llap4585
# @Project: T5-Refiner-DomainFocus
# @License: Apache-2.0
# @GitHub: https://github.com/llap4585/T5-Refiner-DomainFocus

import random
import os
import json
import regex as re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Parameter Configuration ---
NOVEL_FILE_PATH = '.txt'           # Path to the input novel or text file
KEYWORDS_FILE_PATH = 'keywords.txt'    # Path to the keyword list file
OUTPUT_FILE_PATH = 't5_training_set.jsonl'  # Path to save the generated T5 training dataset（jsonl）

MIN_PARAGRAPHS = 10                    # Minimum number of paragraphs per text chunk
MAX_CHAR_LENGTH = 440                   # Maximum number of characters per text chunk(<512 tokens)

KEYWORD_MASK_PROBABILITY = 0.5         # Probability to mask a keyword(50%-70%)
TOTAL_MASK_RATIO = 0.20                # Overall ratio of tokens to mask(15%-30%)
NORMAL_MASK_MAX_LENGTH = 3             # Maximum length of a randomly masked span(2-5)

MAX_WORKERS = 12                        # Maximum number of threads for parallel sample generation


CONTENT_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')# Regex to identify content characters (Chinese, English letters, numbers)

SENTENCE_SPLIT_RE = re.compile(r'([。！？；，,”\.\!\?\\n])')  # Can be adjusted for different text scenarios

# --- Your T5 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(" ", use_fast=True)  # Load your T5 tokenizer here

def split_sentences(text):
    # Use global precompiled regex for splitting
    sentences = SENTENCE_SPLIT_RE.split(text)
    # Optionally merge sentences with punctuation
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1].strip()
        if sentence:
            result.append(sentence + punctuation)
    return result

def find_unknown_chars(text, tokenizer):
    
    unk_chars = []
    for ch in set(text):
        if tokenizer.unk_token_id in tokenizer(ch)["input_ids"]:
            unk_chars.append(ch)
    return unk_chars

def reverse_tool(lst):
# Keep a small segment of context at the end of the text block

    total = 0
    result=[]
    for item in reversed(lst):
        if total + len(item) < 40:  
            result.append(item)
            
            total += len(item)
        
        else:
            break
    result.reverse()
    return "".join(result)



def load_keywords(filepath):
    
    if not os.path.exists(filepath):
        print(f"⚠️ Keyword file not found: {filepath}")
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        keywords = {line.strip() for line in f if line.strip()}
    print(f"✅ Loaded {len(keywords)} keywords")
    return keywords

def split_sentences1(text):
# Split text into sentences based on periods, question marks, exclamation marks, semicolons, etc.

    sentences = split_sentences(text) #Different text scenarios may require modifications
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1].strip()
        if sentence:
            result.append(sentence + punctuation)
    
    return(reverse_tool(result))
    
def first_add(chunks):
    
    result=[]
    result.append(chunks[0])
    for i in range(1,len(chunks)):

        add=split_sentences1(chunks[i-1])

        result.append(add+chunks[i])

    return result

def split_text_by_punctuation(paragraph_buffer, max_len=440):

    text = "".join(paragraph_buffer)
    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1]
        full_sentence = sentence + punctuation

        if len(current_chunk) + len(full_sentence) <= max_len:
            current_chunk += full_sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = full_sentence

# Handle any remaining text at the end
    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        yield chunk


def create_text_chunks(filepath):
# """Generate text chunks paragraph by paragraph

    if not os.path.exists(filepath):
        print(f"❌ Input file not found: {filepath}")
        return

    paragraph_buffer = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                paragraph_buffer.append(stripped_line)


            while len(paragraph_buffer) >= MIN_PARAGRAPHS:
                chunk_paragraphs = paragraph_buffer[:MIN_PARAGRAPHS]
                current_text = "".join(chunk_paragraphs)

                while len(current_text) > MAX_CHAR_LENGTH and len(chunk_paragraphs) > 1:
                    chunk_paragraphs.pop()
                    current_text = "".join(chunk_paragraphs)
                yield current_text
                paragraph_buffer = paragraph_buffer[len(chunk_paragraphs):]
            

    if paragraph_buffer:
        buffer_final="".join(paragraph_buffer)
        if len(buffer_final) > MAX_CHAR_LENGTH:
            for chunk in split_text_by_punctuation([buffer_final], max_len=MAX_CHAR_LENGTH):
                yield chunk

        else:
           yield "".join(paragraph_buffer)


def create_masked_data(text, keywords):


    tokens = tokenizer.tokenize(text)
    DEBUG_SHOW_TOKENS=False
    if DEBUG_SHOW_TOKENS:
        print(f"\n {text[:60]}...")
        print("Tokens:", tokens)

    masked_indices = set()
    spans_to_mask = []
    

    #unk_chars=''
    #unk_chars = find_unknown_chars(text, tokenizer)
    #if unk_chars:
        #print(f"⚠️: {unk_chars}")
        
# 1. Keyword masking
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            
            start_char, end_char = match.span()
            start_token = len(tokenizer(text[:start_char], add_special_tokens=False).input_ids)
            end_token = len(tokenizer(text[:end_char], add_special_tokens=False).input_ids)
            if not any(i in masked_indices for i in range(start_token, end_token)):
                spans_to_mask.append((start_token, end_token))
                masked_indices.update(range(start_token, end_token))

# 2. Random masking
    num_total_tokens = len(tokens)
    num_to_mask_total = int(num_total_tokens * TOTAL_MASK_RATIO)
    num_already_masked = len(masked_indices)
    num_additional_mask = num_to_mask_total - num_already_masked

    available_indices = [i for i in range(num_total_tokens) if i not in masked_indices]

# Ensure at least one mask per text chunk
    if not masked_indices and available_indices:
        num_additional_mask = max(1, num_additional_mask)

    attempts, masked_count = 0, 0
    while masked_count < num_additional_mask and attempts < num_total_tokens * 2:
        attempts += 1
        if not available_indices:
            break
        span_length = random.randint(1, NORMAL_MASK_MAX_LENGTH)
        start_pos = random.choice(available_indices)

# If a punctuation is selected, break the mask on both sides
        cur_token = tokens[start_pos]
        if re.match(r"[\p{P}\p{S}]", cur_token):
            continue  

        end_pos = start_pos + span_length
        span_tokens = tokens[start_pos:end_pos]
        
        if any(re.match(r"[\p{P}\p{S}]", t) for t in span_tokens):
# Stop the span early if a punctuation is encountered
            for i, t in enumerate(span_tokens):
                if re.match(r"[\p{P}\p{S}]", t):
                    end_pos = start_pos + i
                    break

        spans_to_mask.append((start_pos, end_pos))
        masked_indices.update(range(start_pos, end_pos))
        masked_count += (end_pos - start_pos)
        available_indices = [i for i in available_indices if i not in range(start_pos, end_pos)]

    if not spans_to_mask:
        return None

    spans_to_mask.sort(key=lambda x: x[0])
    merged_spans = []
    for span in spans_to_mask:
        if not merged_spans:
            merged_spans.append(span)
        else:
            last_start, last_end = merged_spans[-1]
            if span[0] <= last_end + 1:
                merged_spans[-1] = (last_start, max(last_end, span[1]))
            else:
                merged_spans.append(span)
    spans_to_mask = merged_spans

# 3. Construct input and output
    input_parts, target_parts = [], []
    mask_token_id = 0
    last_token_idx = 0
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    for start, end in spans_to_mask:
        if end <= start:
            continue
        
# If all tokens in this span are unknown (<unk>), it can be skipped
        span_tokens = token_ids[start:end]
        if all(t == tokenizer.unk_token_id for t in span_tokens):
            continue
        input_parts.extend(token_ids[last_token_idx:start])

        input_parts.append(tokenizer.convert_tokens_to_ids(f"<extra_id_{mask_token_id}>"))

        target_parts.append(tokenizer.convert_tokens_to_ids(f"<extra_id_{mask_token_id}>"))
        target_parts.extend(token_ids[start:end])

        last_token_idx = end
        mask_token_id += 1

    input_parts.extend(token_ids[last_token_idx:])
    input_text = tokenizer.decode(input_parts, skip_special_tokens=False)
    target_text = tokenizer.decode(target_parts, skip_special_tokens=False)

    return input_text, target_text



def main():
    print("🚀 Start generating T5 training data...")

    keywords = load_keywords(KEYWORDS_FILE_PATH)
    chunks = list(create_text_chunks(NOVEL_FILE_PATH))
    chunks = first_add(chunks)
    
    total = len(chunks)
    print(f"📖 Total of {total} text chunks to process")
    results = []
    sample_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(create_masked_data, chunk, keywords): i for i, chunk in enumerate(chunks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples..."):
            result = future.result()
            if result:  
                results.append((futures[future], result))
                sample_count += 1


    results.sort(key=lambda x: x[0])

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        for _, (inp, tgt) in results:
            f.write(json.dumps({"inputs": inp, "targets": tgt}, ensure_ascii=False) + "\n")

    print(f"\n✅ Successfully generated {sample_count} training samples")
    print(f"📁 Saved to: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()