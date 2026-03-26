import torch
import torch.nn as nn

class MeanCurveLoss(nn.Module):
    def __init__(self, num_samples=100, eps=1e-6):
        """
        Initialize the curvature loss function
        :param num_samples: Number of sampling points N in the interval u ∈ [0, 1]
        :param eps: Numerical stability constant to prevent division by zero
        """
        super().__init__()
        self.num_samples = num_samples
        self.eps = eps
        # Pre-compute sampling points u [0, 1]
        self.u_samples = torch.linspace(0, 1, steps=num_samples)

    def _get_bezier_derivatives(self, ctrl_pts, u):
        """
        Calculate first and second derivatives of cubic Bézier curves
        :param ctrl_pts: Control points [Batch, Num_Lanes, 4, 2] (4 control points, x,y coordinates)
        :param u: Sampling parameters [N]
        :return: B_prime, B_double_prime [Batch, Num_Lanes, N, 2]
        """
        B, M, _, _ = ctrl_pts.shape
        
        # Extract control points P0, P1, P2, P3
        P0 = ctrl_pts[:, :, 0, :].unsqueeze(2)  # [B, M, 1, 2]
        P1 = ctrl_pts[:, :, 1, :].unsqueeze(2)
        P2 = ctrl_pts[:, :, 2, :].unsqueeze(2)
        P3 = ctrl_pts[:, :, 3, :].unsqueeze(2)
        
        # Reshape u for broadcasting [1, 1, N, 1]
        u = u.view(1, 1, -1, 1).to(ctrl_pts.device)
        
        # --- Equation (1): First derivative B'(u) ---
        # B'(u) = 3(1-u)^2 (P1 - P0) + 6u(1-u) (P2 - P1) + 3u^2 (P3 - P2)
        term1 = 3 * (1 - u)**2 * (P1 - P0)
        term2 = 6 * u * (1 - u) * (P2 - P1)
        term3 = 3 * u**2 * (P3 - P2)
        B_prime = term1 + term2 + term3  # [B, M, N, 2]
        
        # --- Equation (2): Second derivative B''(u) ---
        # Note: Original text had a typo '6t', corrected to '6u'
        # B''(u) = 6(1-u) (P2 - 2P1 + P0) + 6u (P3 - 2P2 + P1)
        term1_2 = 6 * (1 - u) * (P2 - 2 * P1 + P0)
        term2_2 = 6 * u * (P3 - 2 * P2 + P1)
        B_double_prime = term1_2 + term2_2  # [B, M, N, 2]
        
        return B_prime, B_double_prime

    def _calculate_curvature(self, B_prime, B_double_prime):
        """
        Calculate curvature kappa(u) according to Equations (3)-(6)
        :param B_prime: First derivative [B, M, N, 2]
        :param B_double_prime: Second derivative [B, M, N, 2]
        :return: curvature [B, M, N]
        """
        # Extract x, y components
        x1 = B_prime[..., 0]
        y1 = B_prime[..., 1]
        x2 = B_double_prime[..., 0]
        y2 = B_double_prime[..., 1]
        
        # --- Equation (6): Magnitude of 2D cross product |x1*y2 - y1*x2| ---
        cross_product_mag = torch.abs(x1 * y2 - y1 * x2)
        
        # --- Equation (4): Magnitude of first derivative ||B'(u)|| ---
        norm_prime = torch.norm(B_prime, dim=-1)  # [B, M, N]
        
        # --- Equation (3): Curvature kappa = |cross| / ||B'||^3 ---
        # Add eps to prevent division by zero (derivative magnitude may approach 0 for straight lines)
        curvature = cross_product_mag / (torch.pow(norm_prime, 3) + self.eps)
        
        return curvature

    def forward(self, pred_ctrl_pts, gt_ctrl_pts):
        """
        Calculate Mean Curve Loss (Equation 7)
        :param pred_ctrl_pts: Predicted control points [B, M, 4, 2]
        :param gt_ctrl_pts: Ground truth control points [B, M, 4, 2]
        :return: L_mc scalar
        """
        # Ensure u_samples are on the correct device
        u = self.u_samples.to(pred_ctrl_pts.device)
        
        # 1. Calculate derivatives and curvature for predicted curves
        pred_b_prime, pred_b_double = self._get_bezier_derivatives(pred_ctrl_pts, u)
        pred_kappa = self._calculate_curvature(pred_b_prime, pred_b_double)
        
        # 2. Calculate derivatives and curvature for ground truth curves
        gt_b_prime, gt_b_double = self._get_bezier_derivatives(gt_ctrl_pts, u)
        gt_kappa = self._calculate_curvature(gt_b_prime, gt_b_double)
        
        # 3. Equation (7): L_mc = 1/N * sum(|kappa_pred - kappa_gt|)
        # Using L1 loss here, MSELoss can also be used
        diff = torch.abs(pred_kappa - gt_kappa)
        loss_mc = torch.mean(diff)
        
        return loss_mc

class Loss(nn.Module):
    """
    Combined loss function for regression and curvature losses (Equation 8)
    L_lane = lambda * L_reg + (1 - lambda) * L_mc
    """
    def __init__(self, lambda_reg=0.5, num_samples=100):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.curve_loss = MeanCurveLoss(num_samples=num_samples)
        # Example regression loss, can be replaced with L1/L2 based on actual needs
        self.reg_loss = nn.SmoothL1Loss() 

    def forward(self, pred_ctrl_pts, gt_ctrl_pts, pred_exists=None, gt_exists=None):
        # 1. Calculate regression loss (control point coordinate differences)
        # Note: In actual use, invalid lanes may need to be filtered based on existence masks
        loss_reg = self.reg_loss(pred_ctrl_pts, gt_ctrl_pts)
        
        # 2. Calculate curvature loss
        loss_mc = self.curve_loss(pred_ctrl_pts, gt_ctrl_pts)
        
        # 3. Equation (8): Weighted combination
        loss_lane = self.lambda_reg * loss_reg + (1 - self.lambda_reg) * loss_mc
        
        return loss_lane, loss_reg, loss_mc

# --- Usage Example ---
if __name__ == "__main__":
    # Simulate  Batch=2, Num_Lanes=7, 4 control points, 2D coordinates (x,y)
    batch_size = 2
    num_lanes = 7
    pred_ctrl_pts = torch.randn(batch_size, num_lanes, 4, 2, requires_grad=True)
    gt_ctrl_pts = torch.randn(batch_size, num_lanes, 4, 2)
    
    # Initialize loss
    criterion = CombinedLaneLoss(lambda_reg=0.7, num_samples=50)
    
    # Forward pass
    total_loss, loss_reg, loss_mc = criterion(pred_ctrl_pts, gt_ctrl_pts)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Reg Loss: {loss_reg.item():.4f}")
    print(f"Curve Loss: {loss_mc.item():.4f}")
    
    # Backward pass
    total_loss.backward()
