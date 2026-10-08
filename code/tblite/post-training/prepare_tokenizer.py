"""Combine pinned Qwen3 tokenizer assets with the training assistant-mask template."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='Qwen/Qwen3-4B');p.add_argument('--revision',default='1cfa9a7208912126459214e8b04321603b3df60c');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Output must not exist')
 from transformers import AutoTokenizer
 ref=Path(__file__).resolve().parents[1]/'selection/recipe/tokenizer_template'
 cfg=json.loads((ref/'tokenizer_config.json').read_text())
 template=(ref/'chat_template.jinja').read_text() if (ref/'chat_template.jinja').exists() else cfg.get('chat_template')
 if not template:raise ValueError('Training chat template is missing')
 tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision);tok.chat_template=template;tok.save_pretrained(a.output)
 print('Saved training tokenizer to',a.output)
if __name__=='__main__':main()
