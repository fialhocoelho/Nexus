import torch
import torch.nn as nn

# Define student model (smaller model)
class StudentModel(nn.Module):
    """
    Student Model class for sequence prediction.

    Attributes:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the LSTM layer.
        output_size (int): Size of the output.

    Methods:
        forward(x): Forward pass of the model.
        init_hidden(batch_size): Initialize hidden state of the LSTM.
    """
    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 #beta = 0.5,
                 #distillation=False
                 ):
        """
        Initialize StudentModel.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state in the LSTM layer.
            output_size (int): Size of the output.
        """
        super(StudentModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Get the output from the last timestep
        output = self.fc(lstm_out)
        return output

    def init_hidden(self, batch_size):
        """
        Initialize hidden state of the LSTM.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing hidden states.
        """
        return (torch.zeros(1, batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.hidden_size).to(self.device))