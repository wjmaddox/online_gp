import unittest
from online_gp.datasets.classification import Banana
from online_gp.models.online_svgp_classifier import OnlineSVGPClassifier
from online_gp.models.stems import Identity
import torch


class TestOnlineSVGPClassifier(unittest.TestCase):
    def test_batch_classification(self):
        datasets = Banana()
        train_dataset, test_dataset = datasets.train_dataset, datasets.test_dataset
        train_x, train_y = train_dataset[:]
        test_x, test_y = test_dataset[:]

        test_x = test_x / train_x.abs().max(0)[0]
        train_x = train_x / train_x.abs().max(0)[0]

        input_dim = train_x.size(-1)
        stem = Identity(input_dim)
        num_inducing = 128
        lr = 1e-2

        classifier = OnlineSVGPClassifier(stem, train_x, num_inducing, lr, streaming=False)
        if torch.cuda.is_available():
            classifier = classifier.cuda()

        classifier.fit(train_x, train_y, num_epochs=100)

        test_pred = classifier.predict(test_x)
        test_acc = test_pred.eq(test_y).float().mean()
        self.assertGreaterEqual(test_acc, 0.85)

    def test_online_classification(self):
        datasets = Banana()
        train_dataset, test_dataset = datasets.train_dataset, datasets.test_dataset
        train_x, train_y = train_dataset[:]
        test_x, test_y = test_dataset[:]
        test_x = test_x / train_x.abs().max(0)[0]
        train_x = train_x / train_x.abs().max(0)[0]

        num_train, input_dim = train_x.shape
        stem = Identity(input_dim)
        num_inducing = 128
        lr = 1e-2
        beta = 1e-3
        num_update_steps = 1
        batch_size = 1
        train_x = torch.chunk(train_x, train_x.size(0) // batch_size)
        train_y = torch.chunk(train_y, train_y.size(0) // batch_size)

        classifier = OnlineSVGPClassifier(stem, train_x[0], num_inducing, lr, streaming=True, beta=beta,
                                          num_update_steps=num_update_steps, learn_inducing_locations=True)
        if torch.cuda.is_available():
            classifier = classifier.cuda()

        correct = 0
        for t, (x, y) in enumerate(zip(train_x, train_y)):
            pred_y = classifier.predict(x)
            classifier.update(x, y)
            correct += pred_y.eq(y).sum().float().item()
        cum_acc = correct / num_train
        self.assertGreaterEqual(cum_acc, 0.65)

        test_pred = classifier.predict(test_x)
        test_acc = test_pred.eq(test_y).float().mean()
        self.assertGreaterEqual(test_acc, 0.75)


if __name__ == '__main__':
    unittest.main()
