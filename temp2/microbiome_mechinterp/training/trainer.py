import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from ..models.vae import VAELoss


class MicrobiomeVAETrainer:
    """Trainer for VAE models with robust NaN handling and early stopping"""

    def __init__(self, model, device, lr=1e-4):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.model.to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly to avoid NaN losses"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train_epoch(self, train_loader, beta=0.01, alpha=0.001, noise_factor=0.01):
        """Train one epoch with robust NaN handling"""
        self.model.train()
        total_loss = 0
        valid_batches = 0

        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(self.device)

            if torch.isnan(data).any() or torch.isinf(data).any():
                continue

            noisy_data = data + torch.randn_like(data) * noise_factor * 0.1

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(noisy_data)

            if (
                torch.isnan(recon_batch).any()
                or torch.isinf(recon_batch).any()
                or torch.isnan(mu).any()
                or torch.isinf(mu).any()
                or torch.isnan(logvar).any()
                or torch.isinf(logvar).any()
            ):
                continue

            loss, recon_loss, kl_loss, l1_loss = VAELoss(
                recon_batch, data, mu, logvar, beta, alpha
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            valid_grads = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_grads = False
                        break

            if not valid_grads:
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            print("Warning: No valid batches in this epoch!")
            return float("inf")

        return total_loss / valid_batches

    def validate(self, val_loader, beta=0.01, alpha=0.001):
        """Validate with robust NaN handling"""
        self.model.eval()
        total_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for data in val_loader:
                data = data[0].to(self.device)

                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue

                recon_batch, mu, logvar = self.model(data)

                if (
                    torch.isnan(recon_batch).any()
                    or torch.isinf(recon_batch).any()
                    or torch.isnan(mu).any()
                    or torch.isinf(mu).any()
                    or torch.isnan(logvar).any()
                    or torch.isinf(logvar).any()
                ):
                    continue

                loss, _, _, _ = VAELoss(recon_batch, data, mu, logvar, beta, alpha)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1

        if valid_batches == 0:
            return float("inf")

        return total_loss / valid_batches

    def train(
        self,
        data,
        epochs=100,
        batch_size=32,
        learning_rate=None,
        noise_factor=0.01,
        val_data=None,
        early_stopping=True,
        patience=15,
        min_delta=0.0001,
        beta=0.01,
        alpha=0.001,
    ):
        """Train VAE with conservative parameters and robust NaN handling"""

        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)

        train_dataset = TensorDataset(data_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            if isinstance(val_data, np.ndarray):
                val_tensor = torch.FloatTensor(val_data).to(self.device)
            else:
                val_tensor = val_data.to(self.device)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        model_saved = False

        print(
            f"Training VAE on {data.shape[0]} samples with conservative parameters..."
        )
        print(f"  beta={beta}, alpha={alpha}, noise_factor={noise_factor}")
        if early_stopping and val_loader is not None:
            print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, beta, alpha, noise_factor)

            if train_loss == float("inf"):
                print(f"Infinite loss at epoch {epoch}, stopping training")
                break

            train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader, beta, alpha)

                if val_loss == float("inf"):
                    print(
                        f"Infinite validation loss at epoch {epoch}, stopping training"
                    )
                    break

                val_losses.append(val_loss)
            else:
                val_loss = None

            if epoch % 10 == 0 or epoch == 1:
                if val_loader is not None:
                    print(
                        f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
                else:
                    print(f"Epoch {epoch}/{epochs}: Loss: {train_loss:.6f}")

            if early_stopping and val_loader is not None:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    model_path = f"best_vae_model_{self.model.input_dim}.pt"
                    torch.save(self.model.state_dict(), model_path)
                    model_saved = True

                    if epoch % 10 == 0 or epoch == 1:
                        print(
                            f"  → New best model! Val loss improved to {best_val_loss:.6f}"
                        )
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(
                        f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}"
                    )
                    print(f"No improvement for {patience} consecutive epochs")
                    break

        if model_saved:
            model_path = f"best_vae_model_{self.model.input_dim}.pt"
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print(f"Restored model from epoch {best_epoch}")
            else:
                print("Warning: Best model file not found, using current weights")
        else:
            print("Warning: No model was saved during training")

        training_time = time.time() - start_time

        if early_stopping and val_loader is not None:
            if patience_counter < patience and len(train_losses) == epochs:
                print(
                    f"\nTraining completed all {epochs} epochs in {training_time:.2f}s"
                )
                print(
                    f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}"
                )
            elif patience_counter >= patience:
                print(f"Training completed in {training_time:.2f}s")
        else:
            print(f"\nTraining completed in {training_time:.2f} seconds")

        return train_losses, val_losses, best_epoch

    def get_latent_representation(self, data):
        """Get latent representation of data"""
        self.model.eval()

        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)

        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=64)

        latent_vectors = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                mu, _ = self.model.encode(batch_data)
                latent_vectors.append(mu.cpu().numpy())

        return np.vstack(latent_vectors)

    def save_checkpoint(self, filepath, epoch, train_loss, val_loss=None):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
        if checkpoint["val_loss"] is not None:
            print(f"  Val Loss: {checkpoint['val_loss']:.6f}")

        return checkpoint
