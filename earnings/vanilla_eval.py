import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
import whisper

import sys
sys.path.append('../')
from lib import get_path

earnings_base_path = get_path('earnings_base_path')

TEST_PATH = os.path.join(earnings_base_path, 'test_original')
DEV_PATH = os.path.join(earnings_base_path, 'dev_original')
ALL_TEXT_PATH = os.path.join(earnings_base_path, 'full_transcripts.json')

def fetch_data(audio_path:str = TEST_PATH, txt_path:str = ALL_TEXT_PATH):
    with open(txt_path, 'r') as f:
        all_text_json = json.load(f)

    audio_files = [{
        'meeting': el.replace('.mp3', ''),
        'path': os.path.join(audio_path, el)
        } for el in os.listdir(audio_path) if el.endswith('.mp3')]

    text_files = [{
        'meeting': el['meeting'],
        'text': all_text_json[el['meeting']]
        } for el in audio_files]
 

    return audio_files, text_files

def preprocess_transcript(text:str):
    text = text.lower()
    text = text.replace('<silence>', '')
    text = text.replace('<inaudible>', '')
    text = text.replace('<laugh>', '')
    text = text.replace('<noise>', '')
    text = text.replace('<affirmative>', '')
    text = text.replace('<crosstalk>', '')    
    text = text.replace('…', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = re.sub(' +', ' ', text)
    return normalize(text).lower()


def main(args):
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH

    model = whisper.load_model(args.model)
    
    audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
    meetings_keys = [el['meeting'] for el in audio_files]

    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}')

        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['meeting'] and audio_files[rec]['meeting'] == text_files[rec]['meeting'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["meeting"]}, {audio_files[rec]["meeting"]}'

        result = model.transcribe(cur_audio)
        out_text = result['text']
        out_text = normalize(out_text).lower()
        print(cur_text, '\n', out_text, '\n\n')

        all_texts.append(out_text)
        all_golds.append(cur_text)

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)
    print(f'WER: {wer}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='base.en', help='Whisper model name')

    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')

    args = parser.parse_args()
    main(args)