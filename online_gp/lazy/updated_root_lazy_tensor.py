import torch
from gpytorch.lazy import lazify, LazyTensor, RootLazyTensor
from gpytorch.settings import cholesky_jitter
from gpytorch.utils.memoize import cached
from gpytorch.utils.cholesky import psd_safe_cholesky
import warnings

# from ..utils.timing import timed

class UpdatedRootLazyTensor(LazyTensor):
    def __init__(
        self,
        initial_tensor=None,
        n_shape=None,
        initial_is_root=True,
        root=None,
        inv_root=None,
    ):
        r"""
        tensor: initial matrix
        n_shape: shape of matrix if we're building from zero rows
        root: initial root decomposition
        inv_root: initial root inverse decomposition
        """
        if initial_tensor is None:
            initial_tensor = torch.zeros(1, n_shape)

        if initial_is_root:
            initial_tensor = initial_tensor.transpose(-1, -2) @ initial_tensor

        super(UpdatedRootLazyTensor, self).__init__(
            initial_tensor,
            n_shape=n_shape,
            initial_is_root=False,
            root=root,
            inv_root=inv_root,
        )

        self.root = root
        self.inv_root = inv_root

        self.tensor = initial_tensor

    def _size(self):
        return self.tensor.shape

    def _matmul(self, rhs):
        return self.tensor.matmul(rhs)

    def _transpose_nonbatch(self):
        return self

    def update(self, vector):
        if len(vector.shape) == 1:
            vector = vector.view(-1, 1)

        # first update the tensor
        tensor = self.tensor + vector @ vector.transpose(-1, -2)

        # then update the root decomposition
        root, inv_root = self.collect_vector(vector)
        return UpdatedRootLazyTensor(
            tensor,
            initial_is_root=False,
            root=root,
            inv_root=inv_root,
        )

    def collect_vector(self, vector):
        # rank q updates to a root decomposition by storing both the root and its inverse
        # \tilde{A} = A + vv' = L(I + L^{-1} v v' L^{-T})L'

        # first get LL' = A
        current_root = self.root_decomposition().root
        # and BB' = A^{-1}
        current_inv_root = self.root_inv_decomposition().root.transpose(-1,-2)
        
        # compute p = B v and take its SVD
        pvector = current_inv_root.matmul(vector)
        # USV' = p
        # when p is a vector this saves us the trouble of computing an orthonormal basis
        U, S, _ = torch.svd(pvector, some=False)

        # we want the root decomposition of I_r + U S^2 U' but S is q so we need to pad.
        one_padding = torch.ones(torch.Size((*S.shape[:-1], U.shape[-2] - S.shape[-1])), device=S.device, dtype=S.dtype)
        # the non zero eigenvalues get updated by S^2 + 1, so we take the square root.
        root_S_plus_identity = (S**2 + 1.)**0.5
        # pad the nonzero eigenvalues with the ones
        #######
        # \tilde{S} = ((S^2 + 1)^0.5; 0
        # (0; 1)
        #######
        stacked_root_S = torch.cat(
            (root_S_plus_identity, one_padding), dim=-1
        )
        # compute U \tilde{S} for the new root
        inner_root = U.matmul(torch.diag_embed(stacked_root_S))
        # \tilde{L} = L U \tilde{S}
        if inner_root.shape[-1] == current_root.shape[-1]:
            updated_root = current_root.matmul(inner_root)
        else:
            updated_root = torch.cat(
                (
                    current_root.evaluate(), 
                    torch.zeros(*current_root.shape[:-1], 1, device=current_root.device, dtype=current_root.dtype), 
                ),
                dim=-1
            )

        # compute \tilde{S}^{-1}
        stacked_inv_root_S = torch.cat(
            (1./root_S_plus_identity, one_padding), dim=-1
        )
        # compute the new inverse inner root: U \tilde{S}^{-1}
        inner_inv_root = U.matmul(torch.diag_embed(stacked_inv_root_S))
        # finally \tilde{L}^{-1} = L^{-1} U \tilde{S}^{-1}
        updated_inv_root = current_inv_root.transpose(-1, -2).matmul(inner_inv_root)

        return updated_root, updated_inv_root

    @cached(name="root_decomposition")
    def root_decomposition(self, **kwargs):
        if self.root is None:
            return super().root_decomposition(**kwargs)
        else:
            return RootLazyTensor(self.root)

    @cached(name="root_inv_decomposition")
    def root_inv_decomposition(self, **kwargs):
        if self.inv_root is None:
            return super().root_inv_decomposition(**kwargs)
        else:
            return RootLazyTensor(self.inv_root)

    def evaluate(self):
        # evaluation is straightforward, just return the tensor
        return self.tensor

    def _expand_batch(self, batch_shape):
        """
        Expands along batch dimensions.
        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.expand`,
            which does some additional work. Calling this method directly is discouraged.
        """
        current_shape = torch.Size([1 for _ in range(len(batch_shape) - self.dim() + 2)] + list(self.batch_shape))
        batch_repeat = torch.Size(
            [expand_size // current_size for expand_size, current_size in zip(batch_shape, current_shape)]
        )
        #return self.repeat(*batch_repeat, 1, 1)
        # a bit more memory intensive?
        expanded_tensor = self.tensor.repeat(*batch_repeat, 1, 1)
        if self.root is not None:
            expanded_root = self.root.repeat(*batch_repeat, 1, 1)
            expanded_inv_root = self.inv_root.repeat(*batch_repeat, 1, 1)
        
            return UpdatedRootLazyTensor(expanded_tensor, initial_is_root=False, root=expanded_root, inv_root=expanded_inv_root)
        else:
            return UpdatedRootLazyTensor(expanded_tensor, initial_is_root=False)
