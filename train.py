import pickle
import time

from collections import Counter
from Text_RNN import Text_RNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class train:
    def __init__(self, file, name_model= 'model', time=5):  # текстовый файл, папка в которую будет сохраняться модель, время обучения в минутах
        self.device = None
        self.SEQ_LEN = None
        self.BATCH_SIZE = None
        self.name_model = name_model
        self.time = time

        with open(file, "r", encoding="utf-8") as f:
            put = f.read()

        self.text = ' '.join(["".join([g for g in i.lower() if g in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя']) for i in put.split()])

    def tr(self):

        self.SEQ_LEN = 200
        self.BATCH_SIZE = 8

        sequence, char_to_idx, idx_to_char = text_to_seq(self.text)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = Text_RNN(self.device, input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )

        loss_avg = []

        tic = time.perf_counter()

        while 1:
            model.train()
            train, target = self.get_batch(sequence)
            train = train.permute(1, 0, 2).to(self.device)
            target = target.permute(1, 0, 2).to(self.device)
            hidden = model.init_hidden(self.BATCH_SIZE)

            output, hidden = model(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')
                scheduler.step(mean_loss)
                loss_avg = []
                model.eval()
                predicted_text = self.generator(model, char_to_idx, idx_to_char)
                print(predicted_text)
                toc = time.perf_counter()
                if (toc - tic) >= 120:
                    print(f"Прошло {(toc - tic) / 60:0.4f} минут")
                else:
                    print(f"Прошло {toc - tic:0.4f} секунд")
                if mean_loss < 0.8 or (toc - tic) / 60 >= self.time:
                    break

        model.eval()

        torch.save(model.state_dict(), self.name_model + r'/model.pt')
        with open(self.name_model + r'/char_to_idx.pickle', 'wb') as f:
            pickle.dump(char_to_idx, f)
        with open(self.name_model + r'/idx_to_char.pickle', 'wb') as f:
            pickle.dump(idx_to_char, f)

    def get_batch(self, sequence):
        trains = []
        targets = []
        for _ in range(self.BATCH_SIZE):
            batch_start = np.random.randint(0, len(sequence) - self.SEQ_LEN)
            chunk = sequence[batch_start: batch_start + self.SEQ_LEN]
            train = torch.LongTensor(chunk[:-1]).view(-1, 1)
            target = torch.LongTensor(chunk[1:]).view(-1, 1)
            trains.append(train)
            targets.append(target)
        return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

    def generator(self, model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
        hidden = model.init_hidden()
        idx_input = [char_to_idx[char] for char in start_text]
        train = torch.LongTensor(idx_input).view(-1, 1, 1).to(self.device)
        text = start_text

        _, hidden = model(train, hidden)

        inp = train[-1].view(-1, 1, 1)

        for i in range(prediction_len):
            output, hidden = model(inp.to(self.device), hidden)
            p_next = F.softmax(output.cpu().data.view(-1) / temp, dim=-1).detach().cpu().data.numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(self.device)
            predicted_char = idx_to_char[top_index]
            text += predicted_char

        return text


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char