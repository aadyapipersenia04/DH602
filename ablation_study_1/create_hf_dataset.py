from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="og_unn", split="train")
print(dataset[0]["text"])
dataset.push_to_hub("shreyasgrampurohit/neutrophil-text-dataset-og")