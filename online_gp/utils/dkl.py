import torch
from torch.nn import Sequential, Linear, ReLU, Tanh
from torch.optim.lr_scheduler import CosineAnnealingLR


class FeatureExtractor(Sequential):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, grid_bound=1):
        self.input_dim = input_dim
        assert grid_bound > 0
        self.grid_bound = grid_bound
        modules = []
        if num_layers == 1:
            modules.append(Linear(input_dim, output_dim))
        else:
            modules.append(Linear(input_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            modules.append(ReLU())
            modules.append(Linear(hidden_dim, hidden_dim))
        modules.append(ReLU())
        modules.append(Linear(hidden_dim, output_dim))
        modules.append(Tanh())
        super().__init__(*modules)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_dim)
        res = super().forward(inputs)
        return self.grid_bound * res

    def set_requires_grad(self, req_grad: bool):
        for param in self.parameters():
            param.requires_grad = req_grad


def pretrain_stem(stem, train_x, train_y, loss_fn, lr, num_epochs, batch_size, **kwargs):
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    model = torch.nn.Sequential(
        stem,
        torch.nn.Linear(stem.output_dim, train_y.size(-1)).to(train_x.device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sched = CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-4)
    records = []
    model.train()
    for epoch in range(num_epochs):
        avg_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            pred_y = model(inputs)
            loss = loss_fn(pred_y, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(dataloader)
        lr_sched.step()
        records.append(dict(train_loss=avg_loss, epoch=epoch + 1))
    model.eval()
    return records
