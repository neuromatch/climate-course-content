def train_model(cnn_model, train_loader, val_loader, optimizer, criterion, num_epochs, patience):
    """Train a CNN model and store weights/gradients per batch

    Args:
        cnn_model (nn.Module): the model to train
        train_loader (DataLoader): dataloader for training data
        val_loader (DataLoader): dataloader for validation data
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.Module): loss function
        num_epochs (int): total number of training epochs
        patience (int): early stopping patience

    Returns:
        tuple: (train_losses, val_losses, weights_history, grads_history)
    """

    # Initialize trackers for losses and history
    train_losses = []
    val_losses = []
    weights_history = []
    grads_history = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Training Started!")
    for epoch in range(num_epochs):
        cnn_model.train()  # Set model to training mode
        train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            Y_pred = cnn_model(X_batch)  # Forward pass-X_batch
            loss = criterion(Y_pred,Y_batch)  # Compute loss between Y_pred  Y_batch
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate batch loss
            train_loss += loss.item()

            # Store weights and gradients for each layer
            weights_history.append([p.clone().detach().cpu().numpy() for p in cnn_model.parameters()])
            grads_history.append([p.grad.clone().detach().cpu().numpy() for p in cnn_model.parameters() if p.grad is not None])

        # Compute average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on validation set
        cnn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                Y_val_pred = cnn_model(X_val)
                val_loss += criterion(Y_val_pred, Y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("Training Complete!")
    return train_losses, val_losses, weights_history, grads_history