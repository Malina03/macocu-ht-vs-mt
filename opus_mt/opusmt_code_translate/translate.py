

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


def real_thing(file_path, out_file):
	dataset = load_dataset("text", data_files={"test": file_path})
	print(f"Dataset: {dataset['test'][0]}")
	print(f"Encoding file...")
	data = dataset.map(lambda x: encode(x, tokenizer), batched=True)
	print(f"Example: {data['test'][0]}")
	data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
	dataloader = torch.utils.data.DataLoader(data['test'], batch_size=batch_size)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.eval().to(device)
	print(f"Translating...")
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

for f in os.listdir(helsinki_in_path):
	if os.path.isfile(os.path.join(helsinki_in_path, f)):
		# files.append(f)
		f_type = f.split("_")[0]
		year = f.split("_")[3].split(".")[0]
		language = f.split("_")[2]
		lang_pair = f.split("_")[1]
		in_fname = os.path.join(helsinki_in_path, f)
		if f_type == "org":
			out_fname = os.path.join(helsinki_out_path,'trans_'+lang_pair+'_en_'+year+'.txt')
		elif f_type == 'trans':
			out_fname = os.path.join(helsinki_out_path, 'org_' + lang_pair + '_en_' + year + '.txt')
		else:
			print("Invalid file name: ", f)
			continue
		# print(in_fname, out_fname)
		real_thing(in_fname, out_fname)
		break

# for f in files:
#for author in authors:
# book_en = "/data/p278972/data/alitra/ennl/datasets/220526/test." + author + ".4eval.en"
# print(author, book_en)
# real_thing(book_en, author)

