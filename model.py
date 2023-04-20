import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def save(self, n_games, best_score, file_name_model='model.pth', file_name_games='n_games'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name_games)

        with open(file_name, 'w') as fileout:
            fileout.write(f'{str(n_games)}\n')
            fileout.write(str(best_score))
            fileout.close()

        file_name = os.path.join(model_folder_path, file_name_model)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name_model='model.pth', file_name_games='n_games'):
        n_games = 1
        best_score = 0

        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name_games)

        if os.path.isfile(file_name):
            with open(file_name, 'r') as filein:
                n_games = filein.readline()
                best_score = filein.readline()
                filein.close()

            file_name = os.path.join(model_folder_path, file_name_model)

            self.load_state_dict(torch.load(file_name))
            self.eval()
            print('Loading existing state dict.')
            return n_games, best_score

        print('No existing state dict found. Starting from scratch.')
        return n_games, best_score


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )  # defines it as a tuple

        # 1: predicted Q values with current state
        pred = self.model(state)                # decides where the snake should go

        target = pred.clone()

        for i in range(len(game_over)):
            Q_new = reward[i]

            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_Q) -> only if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        # print('loss', loss)
        # print('pred', pred)
        # print('target', target)
        # print('state', next_state)
