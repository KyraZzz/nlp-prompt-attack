from transformers import AutoTokenizer
import random

def label_voting(input_ids, k = 1):
    num_trails = len(input_ids)
    label_count_map = {}
    for i in range(num_trails):
        for j, id in enumerate(input_ids[i]):
            label_count_map[id] = label_count_map.get(id, 0) + 1
    l = list(label_count_map.items())
    random.shuffle(l)
    label_count_map = dict(l)
    return sorted(label_count_map, key=label_count_map.get, reverse=True)[:k]
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    input_ids = [
        [32163, 2396, 49133, 5336, 30032, 49795, 49543, 19257, 8010, 48898],
        [46234, 48743, 12799, 49133, 40962, 43163, 40353, 49208, 49600, 49174],
        [48922, 43695, 47159, 47989, 49003, 40076, 33672, 24788, 37418, 46599], 
        [35707, 9431, 48922, 2918, 49706, 19714, 47791, 49360, 13201, 40311],
        [46171, 2971, 48926, 22701, 9724, 5032, 48779, 49204, 40415, 42185]
    ]
    token_ids = label_voting(input_ids, k = 5)
    print(f"token_ids: {token_ids}, tokens:{[tokenizer.convert_ids_to_tokens(x) for x in token_ids]}")