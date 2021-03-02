from .variational_gp_model import VariationalGPModel
from gpytorch.likelihoods import BernoulliLikelihood
import torch
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import VariationalELBO
import gpytorch


class OnlineSVGPClassifier(torch.nn.Module):
    def __init__(
            self,
            stem,
            init_x,
            num_inducing,
            lr,
            streaming=False,
            beta=1.0,
            learn_inducing_locations=True,
            num_update_steps=1,
            **kwargs
        ):
        super().__init__()
        likelihood = BernoulliLikelihood()
        inducing_points = torch.empty(num_inducing, stem.output_dim)
        inducing_points.uniform_(-1, 1)
        mean_module = ZeroMean()
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=stem.output_dim))
        self.gp = VariationalGPModel(inducing_points, mean_module, covar_module, streaming, likelihood,
                                     beta=beta, learn_inducing_locations=learn_inducing_locations)
        self.mll = None
        self.stem = stem
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.num_update_steps = num_update_steps
        self._raw_inputs = [init_x]

    def forward(self, inputs):
        features = self.stem(inputs)
        return self.gp(features)

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        streaming_state = self.gp.streaming
        self.gp.set_streaming(False)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=len(dataset), beta=1.0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, 1e-4)
        records = []
        num_batches = len(dataloader)
        for epoch in range(num_epochs):
            self.train()
            avg_loss = 0
            for input_batch, target_batch in dataloader:
                self.optimizer.zero_grad()
                features = self.stem(input_batch)
                train_dist = self.gp(features)
                loss = -self.mll(train_dist, target_batch)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() / num_batches
            lr_scheduler.step()

            test_acc = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                with torch.no_grad():
                    test_pred = self.predict(test_x)
                test_acc = test_pred.eq(test_y).float().mean().item()

            records.append(dict(train_loss=avg_loss, test_acc=test_acc, epoch=epoch + 1))
        self.gp.set_streaming(streaming_state)
        self.eval()
        return records

    def predict(self, inputs):
        self.eval()
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        pred_dist = self.gp(features)
        pred_dist = self.gp.likelihood(pred_dist)
        pred_labels = pred_dist.mean.ge(0.5)
        return pred_labels

    def update(self, inputs, targets, update_stem=True):
        # if self.gp.streaming:
        #     self.gp.register_streaming_loss()
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1)
        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=inputs.size(0),
                                   beta=self.gp.beta)
        self.train()
        for _ in range(self.num_update_steps):
            self.optimizer.zero_grad()
            features = self._get_features(inputs)
            features = features if update_stem else features.detach()
            train_dist = self.gp(features)
            loss = -self.mll(train_dist, targets)
            loss.backward()
            self.optimizer.step()
        self.eval()
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
        stem_loss = gp_loss = loss.item()
        return stem_loss, gp_loss

    def set_lr(self, gp_lr, stem_lr=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.optimizer = torch.optim.Adam([
            dict(params=self.gp.parameters(), lr=gp_lr),
            dict(params=self.stem.parameters(), lr=stem_lr)
        ])

    def _get_features(self, inputs):
        # update batch norm stats
        inputs = inputs.view(-1, self.stem.input_dim)
        num_inputs = inputs.size(0)
        num_seen = self._raw_inputs[0].size(0)
        batch_size = 1024
        batch_idxs = torch.randint(0, num_seen, (batch_size,))
        input_batch = self._raw_inputs[0][batch_idxs]
        input_batch = torch.cat([inputs, input_batch])
        features = self.stem(input_batch)
        return features[:num_inputs]
