from create_a_model import GPTLanguageModel
import torch

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./dataset/processed_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = GPTLanguageModel().to('cuda')
model.load_state_dict(torch.load('./model_files/recipes.pt'))
model.eval()
# user_input = input("from what do you want to start ? ")

# context = torch.tensor([encode(user_input)], dtype=torch.long, device='cuda')
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

dummy_input = torch.zeros(1, 128, dtype=torch.long).to("cuda")
torch.onnx.export(model, dummy_input, "./model_files/recipes.onnx", export_params=True, opset_version=9)
