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
        [26914, 19916, 30609, 36722, 26305, 43265, 44460, 37749, 35382, 38821],
        [35734, 40894, 35506, 34711, 33758, 30587, 31408, 17958, 11972, 29354],
        [19019, 18684, 26233, 40894, 18561, 31408, 36300, 25959, 39575, 10590],
        [40894, 15258, 17381, 14888, 36099, 28402, 34395, 42991, 36301, 11538],
        [21498, 35501, 28246, 27933, 29848, 14167, 42494, 34003, 41679, 29062]
    ]
    token_ids = label_voting(input_ids, k = 5)
    print(f"token_ids: {token_ids}, tokens:{[tokenizer.convert_ids_to_tokens(x) for x in token_ids]}")