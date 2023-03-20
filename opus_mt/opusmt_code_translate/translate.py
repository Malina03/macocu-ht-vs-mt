

# https://huggingface.co/Helsinki-NLP/opus-mt-en-nl
# module load CUDA/11.3.1

# Derived from Gabriele Sarti's code
# https://colab.research.google.com/drive/1lpBsJq-If7evjY2dPSd5lwEPTVD7e-_X#scrollTo=n_HvKJLDgWGR

import os
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



def test():
	src_texts = ["I am a small frog!", "My tailor is rich."]
	for src_text in src_texts:
		batch = tokenizer(src_text, return_tensors="pt")
		generated_ids = model.generate(**batch)
		output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		print(output[0])



def encode(examples, tokenizer):
	#print(f"Example: {examples[0]}")
	return tokenizer(examples["text"], truncation=True, padding='max_length')


def real_thing(book_en, author):
	dataset = load_dataset("text", data_files={"test": book_en})
	print(f"Dataset: {dataset['test'][0]}")
	print(f"Encoding file...")
	data = dataset.map(lambda x: encode(x, tokenizer), batched=True)
	print(f"Example: {data['test'][0]}")
	data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
	dataloader = torch.utils.data.DataLoader(data['test'], batch_size=batch_size)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.eval().to(device)
	print(f"Translating...")
	out_file = helsinki_out_path + "/test." + author + ".4eval.helsinki.220904.nl"
	print(out_file)
	with open(out_file, 'a') as f:
		for i, batch in enumerate(tqdm(dataloader)):
			batch = {k: v.to(device) for k, v in batch.items()}
			out = model.generate(**batch)
			translations = tokenizer.batch_decode(out.to("cpu"), skip_special_tokens=True)
			if i == 0:
				print(translations[:2])
			for trans in translations:
				f.write(trans + "\n")


# author = sys.argv[1]
# #authors = ["Hemingway", "Golding"]
# #authors = ["Hemingway"]
# #authors = ["Boyne"]


batch_size = 16
model_name = "Helsinki-NLP/opus-mt-de-en"
helsinki_out_path = "/data/s3412768/opus_mt/translated"
helsinki_in_path = "/data/s3412768/opus_mt/original"
torch.cuda.is_available()

print(f"Loading model and tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# test()

files = []
for path, subdirs, files in os.walk(helsinki_in_path):
	for name in files:
		files.append(os.path.join(path, name))
		print(os.path.join(path, name))
# for f in files:
#for author in authors:
# book_en = "/data/p278972/data/alitra/ennl/datasets/220526/test." + author + ".4eval.en"
# print(author, book_en)
# real_thing(book_en, author)

