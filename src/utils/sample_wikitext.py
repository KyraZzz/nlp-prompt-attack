from datasets import Dataset, load_from_disk, concatenate_datasets
import argparse
from tqdm import tqdm

def sample_wikitext(data_path, train_samples, random_seed, save_path):
    assert train_samples > 1
    train_dataset = load_from_disk(data_path)
    train_dataset = train_dataset.shuffle(random_seed)
    pbar = tqdm(total=train_samples)
    cnt = 0
    idx = 0
    samples = []
    while cnt < train_samples:
        sampled_row = Dataset.from_dict(train_dataset[idx:idx+1])
        idx += 1
        if len(sampled_row['text'][0]) >= 50 and len(sampled_row['text'][0]) <= 150:
            cnt += 1
            samples.append(sampled_row)
            pbar.update(1)
    pbar.close()
    sampled_dataset = concatenate_datasets(samples)
    
    # save datasets to disk
    sampled_dataset.save_to_disk(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = None, help = "Data path")
    parser.add_argument("--train_samples", type = int, default = 30000, help = "The number of train samples")
    parser.add_argument("--random_seed", type = int, default = 42, help = "Model seed")
    parser.add_argument("--save_path", type = str, required = True, help = "Save the sampled dataset into a local directory")
    args = parser.parse_args()

    # sample samples from WikiText dataset and save to a local directory
    sample_wikitext(
        data_path = args.data_path, 
        train_samples = args.train_samples, 
        random_seed = args.random_seed,
        save_path = args.save_path
    )
    
    
    