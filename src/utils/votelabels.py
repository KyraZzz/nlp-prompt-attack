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
        [46953, 36343, 11404, 28101, 8338, 26017, 35260, 24483, 40700, 16779],
        [46953, 29688, 28101, 11404, 36343, 7243, 8338, 30483, 40106, 10760],
        [36343, 29688, 28101, 46953, 40766, 5384, 26017, 11404, 8338, 10760],
        [42157, 36343, 29688, 46953, 11404, 45792, 40766, 8338, 28101, 26053],
        [ 5384, 29688, 46953, 28101, 36343, 8338, 17147, 26017, 11404, 7243]
    ]
    token_ids = label_voting(input_ids, k = 5)
    print(f"token_ids: {token_ids}, tokens:{[tokenizer.convert_ids_to_tokens(x) for x in token_ids]}")