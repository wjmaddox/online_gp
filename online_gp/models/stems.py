import torch


class Identity(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self, inputs):
        return inputs

    def parameters(self, **kwargs):
        return [torch.eye(self.input_dim)]

    def modules(self):
        return []


class LinearStem(torch.nn.Sequential):
    def __init__(self, input_dim, feature_dim):
        modules = [
            torch.nn.Linear(input_dim, feature_dim),
            torch.nn.BatchNorm1d(feature_dim, affine=False),
        ]
        super().__init__(*modules)
        self.input_dim = input_dim
        self.output_dim = feature_dim

    def forward(self, input):
        res = super().forward(input)
        return torch.tanh(res / 2)


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, feature_dim, depth, hidden_dims):
        if isinstance(hidden_dims, str):
            hidden_dims = [int(d) for d in hidden_dims.split(',')]

        modules = [
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU()
        ]

        for i in range(1, depth):
            modules.extend([
                torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                torch.nn.ReLU()
            ])

        modules.extend([
            torch.nn.Linear(hidden_dims[-1], feature_dim),
            torch.nn.BatchNorm1d(feature_dim, affine=False, momentum=1e-1),
        ])

        super().__init__(*modules)
        self.input_dim = input_dim
        self.output_dim = feature_dim

    def forward(self, input):
        res = super().forward(input)
        return torch.tanh(res / 2)
