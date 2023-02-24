import torch
import unittest
from emd import EMDLoss


class TestEMDLoss(unittest.TestCase):
    def test_loss(self):
        emd_loss = EMDLoss()
        p1 = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).cuda()
        p2 = torch.tensor(
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=torch.float32
        ).cuda()
        cost = emd_loss(p1, p2)
        loss = torch.sum(cost)
        self.assertAlmostEqual(loss.item(), 0.3, places=4)

    def test_grad(self):
        emd_loss = EMDLoss()
        p1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).cuda()
        p2 = torch.tensor(
            [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], dtype=torch.float32
        ).cuda()
        p1.requires_grad = True
        p2.requires_grad = True
        cost = emd_loss(p1, p2)
        loss = torch.sum(cost)
        loss.backward()
        p1_grad = torch.tensor(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32
        ).cuda()
        p2_grad = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
        ).cuda()
        self.assertTrue(torch.allclose(p1.grad, p1_grad))
        self.assertTrue(torch.allclose(p2.grad, p2_grad))


if __name__ == "__main__":
    unittest.main()
