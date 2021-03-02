from online_gp.models import OnlineExactClassifier
from online_gp.models.stems import Identity
from online_gp.datasets.classification import Banana
import unittest
import torch


class TestOnlineExactClassifier(unittest.TestCase):
    def test_batch_classification(self):
        datasets = Banana()
        train_dataset, test_dataset = datasets.train_dataset, datasets.test_dataset
        train_x, train_y = train_dataset[:]
        test_x, test_y = test_dataset[:]

        input_dim = train_x.size(-1)
        stem = Identity(input_dim)
        alpha_eps = 1e-2
        lr = 1e-1

        classifier = OnlineExactClassifier(stem, train_x, train_y, alpha_eps, lr)
        if torch.cuda.is_available():
            classifier = classifier.cuda()

        classifier.fit(train_x, train_y, 100)

        test_pred = classifier.predict(test_x)
        test_acc = test_pred.eq(test_y).float().mean()
        self.assertGreaterEqual(test_acc, 0.89)

    def test_online_classification(self):
        num_init = 5
        datasets = Banana()
        train_dataset, test_dataset = datasets.train_dataset, datasets.test_dataset
        train_x, train_y = train_dataset[:]
        init_x, train_x = train_x[:num_init], train_x[num_init:]
        init_y, train_y = train_y[:num_init], train_y[num_init:]
        test_x, test_y = test_dataset[:]

        input_dim = train_x.size(-1)
        stem = Identity(input_dim)
        alpha_eps = 1e-2
        lr = 1e-3

        classifier = OnlineExactClassifier(stem, init_x, init_y, alpha_eps, lr)
        if torch.cuda.is_available():
            classifier = classifier.cuda()

        correct = 0
        for t, (x, y) in enumerate(zip(train_x, train_y)):
            pred_y = classifier.predict(x)
            classifier.update(x, y, update_stem=True, update_gp=True)
            if pred_y == y:
                correct += 1
        cum_acc = correct / train_x.size(0)
        self.assertGreaterEqual(cum_acc, 0.80)

        test_pred = classifier.predict(test_x)
        test_acc = test_pred.eq(test_y).float().mean()
        self.assertGreaterEqual(test_acc, 0.89)


if __name__ == 'main':
    unittest.main()

