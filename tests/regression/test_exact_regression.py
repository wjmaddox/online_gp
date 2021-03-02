import unittest
from online_gp.models.stems import Identity
from online_gp.models.online_exact_regression import OnlineExactRegression
import torch


class TestExactRegression(unittest.TestCase):
    def test_batch_regression(self):
        inputs = torch.stack([
            torch.linspace(-1, 1, 500),
            torch.linspace(-1, 1, 500)
        ], dim=-1)
        targets = torch.stack([
          torch.sin(inputs[:, 0]) + torch.cos(inputs[:, 1]),
          torch.sin(inputs[:, 0]) - torch.cos(inputs[:, 1])
        ], dim=-1)
        targets /= 2
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [400, 100])
        train_x, train_y = train_dataset[:]
        test_x, test_y = test_dataset[:]

        input_dim = train_x.size(-1)
        stem = Identity(input_dim)
        lr = 1e-2

        regression = OnlineExactRegression(stem, train_x, train_y, lr)
        if torch.cuda.is_available():
            regression = regression.cuda()

        regression.fit(train_x, train_y, 100)

        rmse, nll = regression.evaluate(test_x, test_y)
        self.assertLessEqual(rmse, 0.03)
        self.assertLessEqual(nll, 2.)

    def test_online_regression(self):
        num_init = 5
        inputs = torch.stack([
            torch.linspace(-1, 1, 500),
            torch.linspace(-1, 1, 500)
        ], dim=-1)
        targets = torch.stack([
            torch.sin(inputs[:, 0]) + torch.cos(inputs[:, 1]),
            torch.sin(inputs[:, 0]) - torch.cos(inputs[:, 1])
        ], dim=-1)
        targets /= 2
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [400, 100])
        train_x, train_y = train_dataset[:]
        init_x, train_x = train_x[:num_init], train_x[num_init:]
        init_y, train_y = train_y[:num_init], train_y[num_init:]
        test_x, test_y = test_dataset[:]

        input_dim = train_x.size(-1)
        stem = Identity(input_dim)
        lr = 1e-3

        regression = OnlineExactRegression(stem, init_x, init_y, lr)
        if torch.cuda.is_available():
            regression = regression.cuda()

        for t, (x, y) in enumerate(zip(train_x, train_y)):
            regression.evaluate(x, y)
            regression.update(x, y, update_stem=True, update_gp=True)

        rmse, nll = regression.evaluate(test_x, test_y)
        self.assertLessEqual(rmse, 0.03)
        self.assertLessEqual(nll, 1.5)


if __name__ == '__main__':
    unittest.main()