import datasets
from datasets import load_from_disk

dataset = load_from_disk("bezier_dataset2")

print(dataset)

dataset.push_to_hub("JosefKuchar/bezier-dataset")

dataset.load_
