
# --- Define the CNN-LSTM Model ---
class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Model for spatiotemporal climate forecasting.

    - CNN (TimeDistributed) extracts spatial features per timestep.
    - LSTM captures temporal dependencies from CNN-extracted features.
    - Fully Connected Layer maps LSTM output to a climate prediction grid.
    """

    def __init__(self):
        """Initialize CNN-LSTM model layers."""
        super(CNN_LSTM, self).__init__()

        # CNN feature extraction applied to each timestep independently
        self.time_distributed_conv = TimeDistributed(nn.Conv2d(in_channels=4, out_channels=20, kernel_size=3, padding=1))
        self.time_distributed_pool = TimeDistributed(nn.AvgPool2d(2))  # Reduces spatial size
        self.time_distributed_global_pool = TimeDistributed(nn.AdaptiveAvgPool2d((1, 1)))  # Compresses feature maps

        # LSTM processes extracted spatial features across time
        self.lstm = nn.LSTM(input_size=20, hidden_size=25, batch_first=True)

        # Fully connected layer maps LSTM outputs to flattened spatial grid (96x144)
        self.fc = nn.Linear(25, 96 * 144)

        # Reshape layer formats output to match spatial map dimensions
        self.reshape = ReshapeLayer((1, 96, 144))

     # Specify the computations performed on the data
    def forward(self, x):
        """
        Defines forward pass for data flow through the CNN-LSTM model.

        Args:
          x (tensor): Input tensor of shape (batch, timesteps, channels, height, width)

        Returns:
          tensor: Output prediction map of shape (batch, 1, 96, 144)
        """
        batch_size, timesteps, C, H, W = x.size()

        # Apply convolution over each timestep
        x = self.time_distributed_conv(x)

        # Reduce spatial resolution
        x = self.time_distributed_pool(x)

        # Compress feature maps to (1,1) spatial dimensions
        x = self.time_distributed_global_pool(x)

        # Remove (1,1) spatial dimensions
        x = x.squeeze(-1).squeeze(-1)  # Now (batch, timesteps, features)

        # Process the sequence with LSTM
        r_out, (h_n, c_n) = self.lstm(x)

        # Use the final LSTM time step
        r_out = r_out[:, -1, :]  # (batch, hidden_size)

        # Map to spatial output
        x = self.fc(r_out)  # (batch, 96*144)

        # Reshape to final map
        x = self.reshape(x)  # (batch, 1, 96, 144)

        return x