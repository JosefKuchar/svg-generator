import datasets
from datasets import load_from_disk

dataset = load_from_disk("bezier_dataset_2")

print(dataset)

dataset.push_to_hub("JosefKuchar/bezier-dataset")
