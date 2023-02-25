import torch
import pytest
from emd import EMDLoss
import numpy as np


class TestEMDLoss1:
    """
    Test the EMD loss in a simple case.
    """

    emd_loss = EMDLoss()
    p1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).cuda()
    p2 = torch.tensor(
        [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], dtype=torch.float32
    ).cuda()
    p1.requires_grad = True
    p2.requires_grad = True
    cost = emd_loss(p1, p2)
    loss = torch.sum(cost)

    def test_loss1(self):
        loss = self.loss.detach().cpu().numpy()
        assert np.allclose(loss, 0.3)

    def test_grad1(self):
        self.loss.backward()
        p1_grad = self.p1.grad.cpu().numpy()
        p2_grad = self.p2.grad.cpu().numpy()
        assert np.allclose(p1_grad, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        assert np.allclose(p2_grad, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
