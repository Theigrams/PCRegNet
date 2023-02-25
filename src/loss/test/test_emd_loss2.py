import torch
import pytest
from emd import EMDLoss
from scipy.optimize import linear_sum_assignment
import numpy as np


def get_emd_loss(input, target):
    """Compute the Earth Mover's Distance between two point sets."""

    cost = torch.norm(input[:, None, :] - target[None, :, :], dim=-1)
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    emd_loss = torch.sum(cost[row_ind, col_ind])

    input_grad = torch.zeros_like(input)
    target_grad = torch.zeros_like(target)
    for i, j in enumerate(col_ind):
        input_grad[i] = (input[i] - target[j]) / cost[i, j]
        target_grad[j] = (target[j] - input[i]) / cost[i, j]

    return emd_loss, input_grad, target_grad


class TestEMDLoss2:
    """
    Test the EMD loss against a scipy implementation.
    """

    emd_loss = EMDLoss()
    p1 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).cuda()
    p2 = torch.tensor(
        [[0.2, 0, 0], [0, 0.4, 0.8], [0, 0.7, 0.4]], dtype=torch.float32
    ).cuda()
    # m = 5
    # p1 = torch.randn(m, 3, dtype=torch.float32).cuda()
    # p2 = torch.randn(m, 3, dtype=torch.float32).cuda()
    p1.requires_grad = True
    p2.requires_grad = True
    cost = emd_loss(p1, p2)
    loss = torch.sum(cost)

    def test_loss2(self):
        loss = self.loss.detach()
        emd_loss_gt, p1_grad_gt, p2_grad_gt = get_emd_loss(self.p1, self.p2)
        loss_gt = torch.sum(emd_loss_gt)
        assert torch.allclose(loss, loss_gt, atol=1e-4)

    def test_grad2(self):
        emd_loss_gt, p1_grad_gt, p2_grad_gt = get_emd_loss(self.p1, self.p2)
        self.loss.backward()
        p1_grad = self.p1.grad.detach()
        p2_grad = self.p2.grad.detach()
        print(f"p1_grad_gt: {p1_grad_gt}")
        print(f"p1_grad: {p1_grad}")
        print(f"difference: {torch.norm(p1_grad_gt - p1_grad)}")
        assert torch.allclose(p1_grad, p1_grad_gt, atol=1e-4)
        assert torch.allclose(p2_grad, p2_grad_gt, atol=1e-4)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
