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
        [40745, 47490, 47226, 35397, 49089, 48868, 7638, 26932, 2396, 16955],
        [48763, 1061, 5674, 515, 39035, 1940, 46585, 45352, 7041, 1713],
        [49349, 48593, 38936, 49067, 49133, 48317, 47989, 48500, 25570, 49649],
        [49492, 50182, 47499, 41328, 47488, 49349, 49710, 1540, 1728, 43710],
        [45974, 47407, 50176, 49349, 50085, 44395, 5461, 45208, 46385, 45625]
    ]
    token_ids = label_voting(input_ids, k = 5)
    print(f"token_ids: {token_ids}, tokens:{[tokenizer.convert_ids_to_tokens(x) for x in token_ids]}")