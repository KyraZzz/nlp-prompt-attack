from transformers import AutoTokenizer

def label_voting(input_ids, k = 1):
    num_trails = len(input_ids)
    label_count_map = {}
    for i in range(num_trails):
        for j, id in enumerate(input_ids[i]):
            label_count_map[id] = label_count_map.get(id, 0) + 1
    return sorted(label_count_map, key=label_count_map.get, reverse=True)[:k]
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    input_ids = [
        [ 3007, 2373, 32650, 6102, 24390, 40367, 28635, 39018, 8686, 38890],
        [32650, 3007, 25698, 23307, 35907, 41465, 33552, 20161, 29401, 25576],
        [ 3007, 32650, 2373, 23307, 25698, 21319, 16168, 31846, 27462, 34478],
        [32650, 3007, 2373, 23307, 20161, 42824, 30091, 24390, 31846, 11355],
        [32650, 3007, 23307, 33552, 25532, 2373, 25698, 20010, 45103, 7311]
    ]
    token_ids = label_voting(input_ids, k = 5)
    print(f"token_ids: {token_ids}, tokens:{[tokenizer.convert_ids_to_tokens(x) for x in token_ids]}")