from dataset import load_dataset
import torch

print("Dataset loading..")

dataset = load_dataset("SimpleStories/SimpleStories")

stories = dataset['train']['story']

print("Dataset Loaded Successfully..")

special_token = [
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<PAD>"
]

vocab = set()

for story in stories:
    vocab.update(story.split())
    
vocab = special_token + sorted(vocab)

stoi = {
    word : i for i, word in enumerate(vocab)
}

itos = {
    i : word for word, i in stoi.items()
}

encoded = []

for story in stories:
    token = ["<BOS>"] + story.split() + ["<EOS>"]
    
    ids = [
        stoi.get(word, stoi['<UNK>'])
        for word in token
    ]
    
    encoded.extend(ids)
    
encoded = torch.tensor(encoded, dtype=torch.long)

torch.save(encoded, "encoded.pt")
torch.save(vocab, "vocab.pt")

print(f"Vocabulary Size = {len(vocab)}")
print(f"Encoded size = {len(encoded)}")

print("Preprocessing is completed")