import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall

from modules.audio import SR, spectogram
from modules.labels import positives
from modules.samples import all_labels, get_alina_sample, load_samples

input_size = 80
hidden_size = 128


class SumRNN(nn.Module):
    def __init__(self):
        super(SumRNN, self).__init__()
        self.rnn = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x, h):
        h = self.rnn(x, h)
        return self.linear(h), h


def train(file_path='/content/drive/MyDrive/alina_clean-2/model'):
    model = SumRNN()
    optimizer = torch.optim.RMSprop(model.parameters())
    loss = nn.CrossEntropyLoss(weight=torch.Tensor([0, 100]))
    model = torch.load(file_path)
    model.eval()

    while True:
        input = torch.Tensor()
        y_true = torch.Tensor()
        for key in all_labels:
            sig, meta = get_alina_sample(key, sr=SR)
            start = meta.get('start', 0)
            end = meta.get('end', int(SR * 10))
            y = []

            if meta['path'] in positives.get('positives'):

                sample = load_samples(meta['path'])[start:end]

                matrix = spectogram(sample)[:80, :]
                y = torch.cat([torch.zeros(matrix.shape[1] - 15), torch.ones(10), torch.zeros(5)])

            else:
                sample = load_samples(meta['path'])[start:end]

                matrix = spectogram(sample)[:80, :]
                y = torch.zeros(matrix.shape[1])

            input = torch.cat((input, matrix), axis=1)
            y_true = torch.cat((y_true, y))

        optimizer.zero_grad()

        h = torch.zeros(1, hidden_size, dtype=torch.float32)

        predictiton = ''
        error = 0

        for t in range(input.shape[1]):
            feature_column = input[:, t]
            feature_column = feature_column / torch.mean(feature_column)
            feature_column = feature_column.view(1, -1)
            y_pred, h = model(feature_column, h)

            predictiton += str(y_pred.argmax())

            true_answer = y_true[t]

            true_answer = torch.Tensor([true_answer])

            error += loss(y_pred, true_answer.long())

        error /= len(y_true)
        error.backward()
        optimizer.step()

        precision = Precision()
        recall = Recall()
        with torch.no_grad():
            input = torch.Tensor()
            y_true = torch.Tensor()

            for key in all_labels:
                sig, meta = get_alina_sample(key, sr=SR)
                start = meta.get('start', 0)
                end = meta.get('end', int(SR * 10))
                y = []

                if meta['path'] in positives.get('positives'):

                    sample = load_samples(meta['path'])[start:end]

                    matrix = spectogram(sample)[:80, :]
                    y = torch.cat([torch.zeros(matrix.shape[1] - 15), torch.ones(10), torch.zeros(5)])

                else:
                    sample = load_samples(meta['path'])[start:end]

                    matrix = spectogram(sample)[:80, :]
                    y = torch.zeros(matrix.shape[1])

                input = torch.cat((input, matrix), axis=1)
                y_true = torch.cat((y_true, y))

            h = torch.zeros(1, hidden_size, dtype=torch.float32)
            # optimizer.zero_grad()

            predictiton = ''
            error = 0

            for t in range(input.shape[1]):
                feature_column = input[:, t]
                feature_column = feature_column / torch.mean(feature_column)
                feature_column = feature_column.view(1, -1)
                y_pred, h = model(feature_column, h)

                predictiton += str(y_pred.argmax())

                true_answer = y_true[t]
                true_answer = torch.Tensor([true_answer])

                precision.update((y_pred, true_answer.long()))
                recall.update((y_pred, true_answer.long()))

            #        # error /= len(answer_array_test)
            #        # error.backward()
            #        # optimizer.step()

        print('epoch ended:')
        print("Precision: ", precision.compute())  # после каждой эпохи
        print("Recall: ", recall.compute())  # после каждой эпохи

        precision.reset()
        recall.reset()
