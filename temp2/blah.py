#!/usr/bin/env python3
"""
Diagnostic Script for VAE Training Issues
Run this to identify where NaN values are coming from
"""

import torch
import torch.nn as nn
import numpy as np
from microbiome_mechinterp.models.vae import VAE, VAELoss
from microbiome_mechinterp.utils import get_device

print("=" * 80)
print("VAE TRAINING DIAGNOSTICS")
print("=" * 80)

# Setup
device = get_device()
print(f"\nDevice: {device}")

# Create dummy data similar to your real data
print("\n1. Testing with clean synthetic data...")
n_samples = 100
n_features = 785  # Your MetaG feature count
synthetic_data = torch.randn(n_samples, n_features) * 0.5
synthetic_data = synthetic_data.to(device)

print(f"   Data shape: {synthetic_data.shape}")
print(f"   Data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
print(f"   Data mean: {synthetic_data.mean():.3f}, std: {synthetic_data.std():.3f}")
print(f"   Contains NaN: {torch.isnan(synthetic_data).any()}")
print(f"   Contains Inf: {torch.isinf(synthetic_data).any()}")

# Create VAE
print("\n2. Creating VAE model...")
vae = VAE(input_dim=n_features, latent_dim=50).to(device)
print(f"   Model created successfully")
print(f"   Total parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Test forward pass
print("\n3. Testing forward pass...")
vae.train()
try:
    recon, mu, logvar = vae(synthetic_data)
    print(f"   ✓ Forward pass successful")
    print(f"   Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
    print(f"   Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"   Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
    print(f"   Recon NaN: {torch.isnan(recon).any()}, Inf: {torch.isinf(recon).any()}")
    print(f"   Mu NaN: {torch.isnan(mu).any()}, Inf: {torch.isinf(mu).any()}")
    print(
        f"   Logvar NaN: {torch.isnan(logvar).any()}, Inf: {torch.isinf(logvar).any()}"
    )
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    exit(1)

# Test loss computation
print("\n4. Testing loss computation...")
try:
    loss, recon_loss, kl_loss, l1_loss = VAELoss(
        recon, synthetic_data, mu, logvar, beta=0.01, alpha=0.001
    )
    print(f"   ✓ Loss computation successful")
    print(f"   Total loss: {loss.item():.6f}")
    print(f"   Recon loss: {recon_loss.item():.6f}")
    print(f"   KL loss: {kl_loss.item():.6f}")
    print(f"   L1 loss: {l1_loss.item():.6f}")
except Exception as e:
    print(f"   ✗ Loss computation failed: {e}")
    exit(1)

# Test backward pass
print("\n5. Testing backward pass...")
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
try:
    optimizer.zero_grad()
    loss.backward()
    print(f"   ✓ Backward pass successful")

    # Check gradients
    grad_norms = []
    nan_grads = 0
    for name, param in vae.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads += 1
                print(f"   ✗ NaN/Inf gradient in {name}")
            grad_norms.append(param.grad.norm().item())

    if nan_grads == 0:
        print(f"   ✓ All gradients valid")
        print(f"   Max gradient norm: {max(grad_norms):.6f}")
        print(f"   Mean gradient norm: {np.mean(grad_norms):.6f}")
    else:
        print(f"   ✗ {nan_grads} parameters have NaN/Inf gradients")

except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    exit(1)

# Test multiple training steps
print("\n6. Testing 10 training steps...")
successful_steps = 0
for step in range(10):
    try:
        optimizer.zero_grad()
        recon, mu, logvar = vae(synthetic_data)
        loss, _, _, _ = VAELoss(
            recon, synthetic_data, mu, logvar, beta=0.01, alpha=0.001
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"   ✗ Step {step+1}: NaN/Inf loss = {loss.item()}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 0.1)
        optimizer.step()
        successful_steps += 1

        if step == 0 or step == 9:
            print(f"   ✓ Step {step+1}: loss = {loss.item():.6f}")
    except Exception as e:
        print(f"   ✗ Step {step+1} failed: {e}")
        break

if successful_steps == 10:
    print(f"   ✓ All 10 steps successful!")
else:
    print(f"   ✗ Only {successful_steps}/10 steps successful")

# Test with RCLR-like data (centered around 0)
print("\n7. Testing with RCLR-like data (centered, potentially extreme values)...")
rclr_data = torch.randn(n_samples, n_features) * 5.0  # Larger variance like RCLR
rclr_data = rclr_data.to(device)

print(f"   Data range: [{rclr_data.min():.3f}, {rclr_data.max():.3f}]")
print(f"   Data std: {rclr_data.std():.3f}")

try:
    vae_test = VAE(input_dim=n_features, latent_dim=50).to(device)
    recon, mu, logvar = vae_test(rclr_data)
    loss, _, _, _ = VAELoss(recon, rclr_data, mu, logvar, beta=0.01, alpha=0.001)
    print(f"   ✓ RCLR-like data: loss = {loss.item():.6f}")
except Exception as e:
    print(f"   ✗ RCLR-like data failed: {e}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print(
    """
If all tests passed:
- Your model architecture is sound
- The issue is likely in your input data
- Check: Data preprocessing, RCLR transformation, scaling

If tests failed:
- Note which step failed
- Check device compatibility (MPS vs CPU)
- Try running on CPU: device = torch.device('cpu')

Common fixes:
1. Add data validation before training
2. Clip extreme values in RCLR data
3. Use CPU instead of MPS if MPS issues persist
4. Increase numerical stability with smaller learning rate
"""
)

print("\nNext steps:")
print("1. Run this script on your actual preprocessed data")
print("2. Check for extreme values or NaN in your RCLR-transformed data")
print("3. Try training on CPU if MPS backend has issues")
print("4. Consider adding data clipping: data = torch.clamp(data, -10, 10)")
