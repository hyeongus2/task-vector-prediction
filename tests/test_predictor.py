import unittest
import torch
from src.tvp.predictor import TrajectoryPredictor


class PredictorTests(unittest.TestCase):
    def test_zero_and_asymptotic_limit(self):
        model = TrajectoryPredictor(2, 3)
        model.A.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        output = model(torch.tensor([0., 1e8]))
        torch.testing.assert_close(output[0], torch.zeros(3))
        torch.testing.assert_close(output[1], model.A.sum(dim=0))

    def test_small_rate_preserves_nonzero_trajectory_and_gradient(self):
        model = TrajectoryPredictor(1, 1)
        model.A.fill_(1.)
        with torch.no_grad():
            model.log_r.fill_(-20.)
        output = model(torch.tensor([1.]))
        self.assertGreater(output.item(), 0.)
        output.sum().backward()
        self.assertTrue(torch.isfinite(model.log_r.grad).all())
        self.assertGreater(model.log_r.grad.item(), 0.)


if __name__ == "__main__":
    unittest.main()
