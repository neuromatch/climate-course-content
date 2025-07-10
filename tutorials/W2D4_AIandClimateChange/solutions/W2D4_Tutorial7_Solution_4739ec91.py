
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

        #################################################
        ## TODO for students: complete the forward pass ##
        # 1. Apply TimeDistributed Conv2D layer to extract features from each timestep.
        # 2. Reduce the spatial dimensions using average pooling.
        # 3. Apply global average pooling to compress each feature map to size (1,1).
        # 4. Squeeze out the singleton spatial dimensions.
        # 5. Pass the resulting sequence of features to the LSTM.
        # 6. Use the last output of the LSTM.
        # 7. Project to flattened grid using the fully connected layer.
        # 8. Reshape the output to (batch, 1, 96, 144) spatial format.
        #################################################

        # Apply convolution over each timestep
        x = self.time_distributed_conv(x)  # <-- Replace ... with input tensor

        # Reduce spatial resolution
        x = self.time_distributed_pool(x)  # <-- Pass the result of previous layer

        # Compress feature maps to (1,1) spatial dimensions
        x = self.time_distributed_global_pool(x)  # <-- Pass the pooled output

        # Remove (1,1) spatial dimensions
        x = x.squeeze(-1).squeeze(-1)  # <-- Output now (batch, timesteps, features)

        # Process the sequence with LSTM
        r_out, (h_n, c_n) = self.lstm(x)  # <-- Feed squeezed output to LSTM

        # Use the final LSTM time step
        r_out = r_out[:, -1, :]  # <-- Take last timestep’s output

        # Map to spatial output
        x = self.fc(r_out)  # <-- Apply FC to final LSTM output

        # Reshape to final map
        x = self.reshape(x)  # <-- Reshape to (batch, 1, 96, 144)

        return x