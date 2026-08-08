import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


class Head(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        
        
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
    def forward(self, x):
        T = x.shape[0]
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        score = q @ k.T 
        score = score / (k.shape[-1] ** 0.5)
        
        mask = torch.tril(torch.ones(T, T))
        
        score = score.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(score, dim=-1)
        
        output = weights @ v
        
        return output
    
class MultiHead(nn.Module):
    def __init__(self, num_head, head_size, embedding_dim) :
        super().__init__()
        
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size) for _ in range(num_head)]
        )
        
        self.proj = nn.Linear(num_head * head_size, embedding_dim)
        
    def forward(self, x):
        output = [head(x) for head in self.heads]
        
        output = torch.cat(output, dim=-1) # why not edition here why only concat
        
        output = self.proj(output)
        
        return output       
    

class FFN(nn.Module):
    def __init__(self, embedding_dim) :
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            
            nn.ReLU(),
            
            nn.Linear(embedding_dim * 4, embedding_dim)
        )        
        
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, embedding_dim, num_head) :
        super().__init__()
        
        head_size = embedding_dim // num_head
        
        self.sa = MultiHead(num_head, head_size, embedding_dim)
        
        self.ffwd = FFN(embedding_dim)
        
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        
    def forward(self, x):
        x  = x +  self.sa(self.ln1(x))
        
        x = x + self.ffwd(self.ln2(x))
        
        return x
        
        
class LMHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.lm = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        output = self.lm(x)
        return output   


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, num_head, num_layer) :
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        
        self.blocks = nn.ModuleList(
            [Block(embedding_dim, num_head) for _ in range(num_layer)]
        )
                
        self.lm = LMHead(embedding_dim, vocab_size)
        
        
    def forward(self, x):
        T = x.shape[0]
        
        token = self.token_embedding(x)
        
        positions = torch.arange(T)
        
        pos = self.position_embedding(positions)
        
        output = token + pos
        
        for bloc in self.blocks:
            output = bloc(output)
                
        logits = self.lm(output)
        
        return logits
    
    def generate(self, x, max_token):
        
        
        for i in range(max_token):
            x = x[-self.block_size:]
            
            logits = self(x)
            
            last_logit = logits[-1]
            
            probs = F.softmax(last_logit, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
                        
            x = torch.cat((x, next_token), dim=0)
            
        return x


text = torch.load("encoded.pt")
vocab = torch.load("vocab.pt")

vocab_size = len(vocab)

stoi = {ch: i for i , ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

encoded = [stoi[c] for c in text]

encoded = torch.tensor(encoded)

x = encoded[:-1]
y = encoded[1:]

embedding_dim = 4
block_size = len(encoded)
num_head = 2
num_layer = 2


model = GPT(vocab_size, embedding_dim, block_size, num_head, num_layer)

optimiser = optim.Adam(
    model.parameters(),
    lr=0.001
)

logits = model(x)

print(logits.shape)
print(y.shape)

for epoch in range(200):
    logits = model(x)
    
    loss = F.cross_entropy(
        logits,
        y
    )
    
    optimiser.zero_grad()
    
    loss.backward()
    
    optimiser.step()
    
    print("Epoch: ", epoch + 1)
    print(f"Loss: {loss.item():.4f}")


while True:


    prompt = torch.tensor([stoi[c] for c in prompt], dtype=torch.long)

    model.eval()

    with torch.no_grad():
        generated = model.generate(prompt, 50)
        
    generated_text = "".join([itos[token.item()] for token in generated])

    print(generated_text)