import torch
import gpytorch


class DirichletGPClassifier(torch.nn.Module):
    def __init__(self, stem, gp, mll, alpha_eps, lr, *args, **kwargs):
        super().__init__()
        self.stem = stem
        self.gp = gp
        self.alpha_eps = alpha_eps
        self.mll = mll
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._target_batch_shape = []

    def _transform_targets(self, targets, alpha_eps):
        num_targets = targets.size(0)
        alpha = alpha_eps * torch.ones(num_targets, 2).to(targets.device)
        alpha[torch.arange(num_targets), targets] += 1
        sigma2_i = torch.log(1. / alpha + 1.)
        transformed_targets = (alpha.log() - 0.5 * sigma2_i)
        return transformed_targets, alpha, sigma2_i

    def predict(self, inputs):
        self.eval()
        pred_dist = self(inputs)
        pred_labels = pred_dist.mean.argmax(0)
        return pred_labels

    def forward(self, inputs):
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        return self.gp(features)

    def fit(self, inputs, targets, num_epochs):
        raise NotImplementedError

    def set_train_data(self, inputs, targets, noise):
        self.gp.set_train_data(inputs, targets, noise)

    def set_lr(self, gp_lr, stem_lr=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.optimizer = torch.optim.Adam([
            dict(params=self.gp.parameters(), lr=gp_lr),
            dict(params=self.stem.parameters(), lr=stem_lr)
        ])
