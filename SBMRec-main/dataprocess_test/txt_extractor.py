import torch
import html
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from timm.models.vision_transformer import vit_base_patch00_clip_224
from get_df import get_df

def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text[0])
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    cleaned_text = cleaned_text.replace('\t', ' ')
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

def generate_text(args, items, features):
    def not_nan(nan):
        return nan == nan
    item_text_list = []
    already_items = set()
    data_df = get_df(args.meta_path)

    for index in tqdm(range(data_df.shape[0]), desc='Generate text'):
        item = data_df['asin'][index]
        if item in items and item not in already_items:
            already_items.add(item)
            text = ''
            for meta_key in features:
                if meta_key in data_df.columns:
                    content = data_df[meta_key][index]
                    if not_nan(content):
                        meta_value = clean_text(content)
                        text += meta_value + ' '
            item_text_list.append((item, text))
    with open(args.txt_path, 'w') as f:
        for i, record in enumerate(item_text_list):
            line = '\t'.join(record)
            f.write(line + '\n')
    return item_text_list

def load_content(args):
    item_text_list = []
    with open(args.txt_path, 'r') as f:
        while True:
            line = f.readline().strip()
            if len(line) == 0: break
            item, text = line.split('\t')
            item_text_list.append((item, text))
    return item_text_list

def txt_extractor(sentences, args, padding_idx=0):
    # 要求sentence必须事先按id排好序，用clip也可以处理
    tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-uncased/vocab.txt')
    config = BertConfig.from_pretrained('./pretrained/bert-base-uncased/config.json')
    bert = BertModel.from_pretrained('./pretrained/bert-base-uncased/pytorch_model.bin', config=config).to(args.device)
    result = []
    with torch.no_grad():
        for _, sentence in tqdm(enumerate(sentences), desc='Text Extracting', total=len(sentences)):
            sentence = '[CLS] ' + sentence
            token_seq = tokenizer.tokenize(sentence)
            idx_seq = tokenizer.convert_tokens_to_ids(token_seq)
            idx_seq_tensor = torch.tensor(idx_seq, dtype=torch.long).to(args.device).view(1, -1)
            output = bert(idx_seq_tensor)
            result.append(output[0][:, 0, :].cpu())
        result.insert(padding_idx, torch.zeros(result[-1].shape, dtype=torch.float))
        txt_emb = torch.cat(result, dim=0)
    torch.save(txt_emb, args.txt_emb)
