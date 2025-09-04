from datasets import load_dataset

test_set = load_dataset("parquet", data_files="/Users/xrickliao/WorkSpaces/DataSets/speechocean762_parquet",split="test")

print(len(test_set))

next(iter(test_set))