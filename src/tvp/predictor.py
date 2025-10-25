# src/tvp/predictor.py
import torch
import torch.nn.functional as F

class TrajectoryPredictor(torch.nn.Module):
    """A PyTorch model to fit the trajectory tau_hat_t = sum_i a_i * (1 - exp(-r_i * t))."""
    def __init__(self, k: int, d: int):
        """
        Initializes the TrajectoryPredictor model.

        Args:
            k (int): The number of exponential terms to sum.
            d (int): The dimensionality of the vector to be predicted.
        """
        super().__init__()
        
        # --- ALTERNATING OPTIMIZATION ---
        # 'A' is not a trainable parameter but a buffer.
        # It will be calculated analytically and updated manually from outside the model.
        # register_buffer ensures it's part of the model's state_dict and moves with .to(device).
        self.register_buffer('A', torch.zeros(k, d))
        
        # 'log_r' is a trainable parameter.
        # The specific initialization is handled by the analyzer.
        # We just need to initialize it with the correct shape.
        self.log_r = torch.nn.Parameter(torch.randn(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            t (torch.Tensor): A 1D tensor of time steps.

        Returns:
            torch.Tensor: A 2D tensor of shape [len(t), d] representing the predicted vectors.
        """
        # Ensures r is always positive
        rates = F.softplus(self.log_r)
        
        # Unsqueeze for broadcasting: t becomes [N, 1], rates become [1, k]
        t = t.unsqueeze(1)
        rates = rates.unsqueeze(0)
        
        exp_term = 1 - torch.exp(-rates * t)
        
        # Matrix multiplication: [N, k] @ [k, d] -> [N, d]
        # This uses the 'A' buffer, which is updated externally during alternating optimization.
        return exp_term @ self.A