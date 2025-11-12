import torch
import torch.nn as nn
from collections import deque


class QNetwork(nn.Module):
    input_size = 10
    output_size = 2

    def __init__(self, input_size, output_size):
        # TODO: [edit here]

        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNAgent:
    nn = None
    memory = None

    def __init__(self, CONFIGS):
        input_size = CONFIGS.WINDOW_SIZE
        output_size = CONFIGS.OUTPUT_SIZE
        self.nn = QNetwork(input_size, output_size)
        self.memory = deque(maxlen=10000)

    def __call__(self, state):
        with torch.no_grad():
            q_values = self.nn(state)
        return torch.argmax(q_values)

    def parameters(self):
        return self.nn.parameters()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def actions(self, dataset):
        actions = []
        for idx, (x, _) in enumerate(dataset):
            with torch.no_grad():
                q = self(x)
                actions.append([idx, q.item()])

        return actions
