import torch
import torch.nn as nn

def to_tensor(data):
    """
    Converts input data to a torch.Tensor.
    
    Args:
        data (torch.Tensor, list, numpy.array): Input data.
    
    Returns:
        torch.Tensor: Converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    else:
        raise TypeError("Input data should be a torch.Tensor, list, or tuple")

def ioa_loss(predictions, targets):
    """
    Index of Agreement (IoA) Loss.
    
    Args:
        predictions (torch.Tensor, list): Predicted values.
        targets (torch.Tensor, list): True values.
    
    Returns:
        torch.Tensor: IoA loss.

    Formulas:
        Index of Agreement (IoA) Loss:
            d = 1 - (sum((P_i - O_i)^2) / sum((|P_i - O_bar| + |O_i - O_bar|)^2))
            IoA Loss = 1 - d
        
    Reference:
        Willmott, C. J. (1981). "On the validation of models". Physical Geography.
    """
    predictions = to_tensor(predictions).view(-1)  # Ensure input is a tensor and flatten
    targets = to_tensor(targets).view(-1)  # Ensure input is a tensor and flatten
    mean_observed = torch.mean(targets)
    numerator = torch.sum((predictions - targets) ** 2)
    denominator = torch.sum((torch.abs(predictions - mean_observed) + torch.abs(targets - mean_observed)) ** 2)
    if denominator == 0:
        return torch.tensor(0.0, requires_grad=True)
    d = 1 - (numerator / denominator)
    return 1 - d

def smape_loss(predictions, targets):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) Loss.
    
    Args:
        predictions (torch.Tensor, list): Predicted values.
        targets (torch.Tensor, list): True values.
    
    Returns:
        torch.Tensor: sMAPE loss.

    Formulas:        
        Symmetric Mean Absolute Percentage Error (sMAPE) Loss:
            sMAPE = (1/N) * sum(2 * |P_i - O_i| / (|P_i| + |O_i|))

    Reference:
        Armstrong, J. S. (1985). "Long-Range Forecasting: From Crystal Ball to Computer". Wiley.
    """
    predictions = to_tensor(predictions).view(-1)  # Ensure input is a tensor and flatten
    targets = to_tensor(targets).view(-1)  # Ensure input is a tensor and flatten
    epsilon = 1e-8  # Small value to avoid division by zero
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets) + epsilon) / 2
    return torch.mean(numerator / denominator)

def mase_loss(predictions, targets):
    """
    Mean Absolute Scaled Error (MASE) Loss.
    
    Args:
        predictions (torch.Tensor, list): Predicted values.
        targets (torch.Tensor, list): True values.
    
    Returns:
        torch.Tensor: MASE loss.

    Formulas:
        Mean Absolute Scaled Error (MASE) Loss:
            MASE = MAE / (1/N) * sum(|O_i - O_(i-1)|)
    
    Reference:
        Hyndman, R. J., & Koehler, A. B. (2006). "Another look at measures of forecast accuracy". International Journal of Forecasting.
    """
    predictions = to_tensor(predictions).view(-1)  # Ensure input is a tensor and flatten
    targets = to_tensor(targets).view(-1)  # Ensure input is a tensor and flatten
    n = targets.size(0)
    mae = torch.mean(torch.abs(predictions - targets))
    naive_forecast = torch.roll(targets, shifts=1)
    naive_forecast[0] = targets[0]  # First value has no previous value, use itself
    mae_naive = torch.mean(torch.abs(targets - naive_forecast))
    return mae / mae_naive

def mse_loss(predictions, targets):
    """
    Mean Squared Error (MSE) Loss.
    
    Args:
        predictions (torch.Tensor, list): Predicted values.
        targets (torch.Tensor, list): True values.
    
    Returns:
        torch.Tensor: MSE loss.

    Formulas:
        Mean Squared Error (MSE) Loss:
            MSE = (1/N) * sum((P_i - O_i)^2)
    """
    predictions = to_tensor(predictions)
    targets = to_tensor(targets)
    return nn.functional.mse_loss(predictions, targets)

def rmse_loss(predictions, targets):
    """
    Root Mean Squared Error (RMSE) Loss.
    
    Args:
        predictions (torch.Tensor, list): Predicted values.
        targets (torch.Tensor, list): True values.
    
    Returns:
        torch.Tensor: RMSE loss.

    Formulas:
        Root Mean Squared Error (RMSE) Loss:
            RMSE = sqrt((1/N) * sum((P_i - O_i)^2))
    """
    predictions = to_tensor(predictions)
    targets = to_tensor(targets)
    mse = nn.functional.mse_loss(predictions, targets)
    return torch.sqrt(mse)

class DistillationLoss(nn.Module):
    """
    A custom loss function for model distillation that combines the student model's loss with the distillation loss
    from the teacher model's outputs. The loss function can be chosen from a variety of common loss functions.
    
    Args:
        loss_function (function): The loss function to use (e.g., ioa_loss, smape_loss, mase_loss, mse_loss, rmse_loss).
        beta (float): The weight for the distillation loss, balancing between student loss and distillation loss.
    """
    def __init__(self, loss_function, beta=0.5):
        super(DistillationLoss, self).__init__()
        self.loss_function = loss_function
        self.beta = beta

    def forward(self, y, student_outputs, teacher_outputs):
        """
        Compute the combined loss for model distillation.
        
        Args:
            y (torch.Tensor, list): True values.
            student_outputs (torch.Tensor, list): Student model predictions.
            teacher_outputs (torch.Tensor, list): Teacher model predictions.
        
        Returns:
            torch.Tensor: Combined loss.
        """
        y = to_tensor(y)  # Convert to tensor if necessary
        student_outputs = to_tensor(student_outputs)  # Convert to tensor if necessary
        teacher_outputs = to_tensor(teacher_outputs)  # Convert to tensor if necessary

        # Calculate the student loss with respect to the true values
        student_loss = self.loss_function(student_outputs, y)
        
        # Calculate the distillation loss between student and teacher outputs
        distillation_loss = self.loss_function(student_outputs, teacher_outputs)
        
        # Combine the losses using the beta weight
        loss = (1 - self.beta) * student_loss + self.beta * distillation_loss
        
        return loss  # Return the combined loss