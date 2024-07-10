import os
import numpy as np
import time
import torch
from torch import nn
from model import ModelClassifier
from dataset import SkeletonDataset
import matplotlib.pyplot as plt
import pandas as pd

# Define relevant variables for the ML task
num_classes = 10
lr = 0.001
num_epochs = 200

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def traint(model_name: str = 'classifier_v1.0'):
    train_loader = SkeletonDataset(epoch_size=10, batch_size=100)
    model = ModelClassifier()

    loss_fn = nn.MSELoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model = model.to(device)
    model.parameters()
    # model.load_state_dict(torch.load(f"/home/ubuntu/PycharmProjects/scince/FilterAI/checkpoints/{model_name}.pt"))
    # model.eval()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(to_tensor(y_batch).cuda())
            # make_figs(x_batch, y_pred, model_name)
            # exit(0)
            loss = loss_fn(y_pred.cpu(), to_tensor(x_batch))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()

        if epoch % 10 == 0:
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(train_loader):
                y_pred = model(to_tensor(y_batch).cuda())
                val_loss = loss_fn(y_pred.detach().cpu(), to_tensor(x_batch)).item() / len(train_loader)
                avg_val_loss += val_loss

            elapsed_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{num_epochs} \t loss={avg_loss} \t   val_loss={avg_val_loss} \t  time={elapsed_time}s')
    torch.save(model.state_dict(), f'checkpoints/{model_name}.pt')
    make_figs(x_batch, y_pred, model_name)


def make_figs(x, y, name):
    os.makedirs(f"./figs/{name}", exist_ok=True)
    for i in range(30):
        save_fig([x[i], y[i]], f'figs/{name}/{i}.png')


def to_tensor(np_array):
    return torch.from_numpy(np.array(np_array, dtype='float32'))


def save_report_table(report_dict):
    pd.DataFrame.from_dict(report_dict, orient='index', columns=['MSE']).to_excel('table.xlsx')


def save_report_figs(name: str, x: np.ndarray, y: torch.Tensor, labels: list):
    v1 = [x[labels.index(name), :]]
    v2 = [y[labels.index(name), :].detach().cpu().numpy()]
    save_fig(v1, f'x_{name}.png')
    save_fig(v2, f'y_{name}.png')
    save_fig(v1+v2, f'xy_{name}.png')


def save_fig(vectors: list, fname: str):
    plt.gcf().set_size_inches(30, 10)
    plt.clf()
    for vector in vectors:
        if type(vector) != np.ndarray:
            vector = vector.detach().cpu().numpy()
        plt.plot(vector)
    plt.savefig(fname, dpi=300)


def report(train_loader, y_pred):
    y_pred = y_pred.cpu()
    y = train_loader.y
    err = (y - y_pred) ** 2
    mse = (torch.sum(err, axis=1) / y.shape[0]).detach().numpy()
    mse_report = {l: m for l, m in zip(train_loader.labels, mse)}
    mse_report_str = "\n".join([f"{l}: {m}" for l, m in zip(train_loader.labels, mse)])
    return mse_report, mse_report_str


if __name__ == '__main__':
    traint()