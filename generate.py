import pickle
from Text_RNN import Text_RNN
import torch
import torch.nn.functional
import numpy as np


class generate:
    def __init__(self, mod): # папка с моделью которая будет использоваться для генерации
        self.model = mod

    def gen(self, le, st=""): # длина генерируемого текста, стартовая строка
        with open(self.model + r'/char_to_idx.pickle', 'rb') as f:
            char_to_idx = pickle.load(f)
        with open(self.model + r'/idx_to_char.pickle', 'rb') as f:
            idx_to_char = pickle.load(f)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = Text_RNN(device, input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
        model.to(device)
        model.load_state_dict(torch.load(self.model + r"/model.pt"))
        if st != "":
            st = ' '.join(["".join([g for g in i.lower() if g in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя']) for i in st.split()])

        hidden = model.init_hidden()
        train = torch.LongTensor([char_to_idx[char] for char in st]).view(-1, 1, 1).to(device)
        text = st
        _, hidden = model(train, hidden)
        inp = train[-1].view(-1, 1, 1)
        temp = 0.3

        for i in range(le):
            output, hidden = model(inp.to(device), hidden)
            p_next = torch.nn.functional.softmax(output.cpu().data.view(-1) / temp,
                                                 dim=-1).detach().cpu().data.numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            text += idx_to_char[top_index]

        print(text)










