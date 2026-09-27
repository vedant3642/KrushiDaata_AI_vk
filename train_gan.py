import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI errors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Dataset Helper
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 2. Generator Network
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim, num_numerical, cat_dims):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_dims = cat_dims
        
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Numerical output head: output in [0, 1] (matching MinMaxScaler)
        self.num_head = nn.Sequential(
            nn.Linear(256, num_numerical),
            nn.Sigmoid()
        )
        
        # Categorical output heads (one for each categorical field)
        self.cat_heads = nn.ModuleList([
            nn.Linear(256, dim) for dim in cat_dims
        ])
        
    def forward(self, z):
        shared_out = self.shared(z)
        num_out = self.num_head(shared_out)
        
        cat_outs = []
        for head in self.cat_heads:
            # Softmax to produce valid probability distributions
            cat_outs.append(torch.softmax(head(shared_out), dim=-1))
            
        return torch.cat([num_out] + cat_outs, dim=-1)

# ==========================================
# 3. Discriminator (Critic) Network
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. De-processing & Generation Helper
# ==========================================
def generate_synthetic_data(generator, num_samples, scaler, encoder, numerical_cols, categorical_cols, cat_dims, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        synthetic_features = generator(z).cpu().numpy()
        
    # Split back numerical and categorical components
    num_features = synthetic_features[:, :len(numerical_cols)]
    cat_features = synthetic_features[:, len(numerical_cols):]
    
    # Inverse scale numerical columns
    real_numerical = scaler.inverse_transform(num_features)
    
    # Decode one-hot representations back to original categorical strings
    real_categorical = []
    start_idx = 0
    for i, dim in enumerate(cat_dims):
        end_idx = start_idx + dim
        block = cat_features[:, start_idx:end_idx]
        
        # Argmax to determine the generated category index
        class_indices = np.argmax(block, axis=1)
        
        categories = encoder.categories_[i]
        real_categorical.append(categories[class_indices])
        
        start_idx = end_idx
        
    # Build the pandas DataFrame
    df_num = pd.DataFrame(real_numerical, columns=numerical_cols)
    df_cat = pd.DataFrame(np.array(real_categorical).T, columns=categorical_cols)
    
    synthetic_df = pd.concat([df_num, df_cat], axis=1)
    
    # Clip and clean numeric values to maintain physical reality
    for col in ["Nitrogen", "Phosphorus", "Potassium", "Rainfall", "Temperature"]:
        synthetic_df[col] = synthetic_df[col].clip(lower=0).round(2)
    synthetic_df["pH"] = synthetic_df["pH"].clip(lower=0.0, upper=14.0).round(2)
    
    return synthetic_df

# ==========================================
# 5. Plotting / Distribution Comparison
# ==========================================
def plot_distributions(real_df, synthetic_df, save_path="gan_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparison of Real and GAN-Generated Feature Distributions", fontsize=16, fontweight='bold', color='#1e293b')
    
    numerical_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    colors = ['#3b82f6', '#10b981'] # Modern Blue and Emerald Green
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx // 3, idx % 3]
        ax.hist(real_df[col], bins=20, alpha=0.6, label='Real Data', color=colors[0], density=True)
        ax.hist(synthetic_df[col], bins=20, alpha=0.6, label='GAN Synthetic Data', color=colors[1], density=True)
        ax.set_title(col, fontsize=12, fontweight='semibold', color='#334155')
        ax.legend()
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution comparison plot saved to '{save_path}'")

# ==========================================
# 6. Main Pipeline
# ==========================================
def main():
    csv_file = "Crop and fertilizer dataset (1).csv"
    print(f"Loading and preprocessing dataset '{csv_file}'...")
    df = pd.read_csv(csv_file)
    
    # Clean the dataset
    df = df.drop(columns=["Link"], errors="ignore")
    df["Soil_color"] = df["Soil_color"].str.strip()
    df["District_Name"] = df["District_Name"].str.strip()
    df["Crop"] = df["Crop"].str.strip()
    df["Fertilizer"] = df["Fertilizer"].str.strip()
    
    numerical_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    categorical_cols = ["District_Name", "Soil_color", "Crop", "Fertilizer"]
    
    # 6.1 Scaling and Encoding
    scaler = MinMaxScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(df[categorical_cols])
    
    cat_dims = [len(cat) for cat in encoder.categories_]
    print(f"Features setup:")
    print(f"  - Numerical columns: {len(numerical_cols)}")
    print(f"  - Categorical columns encoded dimensions: {cat_dims} (Total size: {sum(cat_dims)})")
    
    # Assemble full tabular matrix
    full_dataset = np.hstack([scaled_numerical, encoded_categorical])
    input_dim = full_dataset.shape[1]
    print(f"  - Combined tabular matrix dimension: {input_dim}")
    
    # 6.2 Data Loader Setup
    batch_size = 64
    epochs = 200
    latent_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = TabularDataset(full_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 6.3 Instantiate GAN Models
    generator = Generator(latent_dim, len(numerical_cols), cat_dims).to(device)
    discriminator = Discriminator(input_dim).to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    # 6.4 Training Loop
    print("\nStarting Tabular GAN Model Training...")
    for epoch in range(epochs):
        for i, real_samples in enumerate(dataloader):
            real_samples = real_samples.to(device)
            
            # --- Train Discriminator ---
            d_optimizer.zero_grad()
            
            # Real samples loss (with soft labels smoothing)
            real_targets = torch.full((batch_size, 1), 0.9, device=device) # Label smoothing
            outputs_real = discriminator(real_samples)
            d_loss_real = criterion(outputs_real, real_targets)
            
            # Fake samples loss
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(z)
            fake_targets = torch.zeros((batch_size, 1), device=device)
            outputs_fake = discriminator(fake_samples.detach())
            d_loss_fake = criterion(outputs_fake, fake_targets)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # --- Train Generator ---
            g_optimizer.zero_grad()
            
            # Generator wants discriminator to think fake samples are real (target label = 1.0)
            g_targets = torch.ones((batch_size, 1), device=device)
            outputs_g = discriminator(fake_samples)
            g_loss = criterion(outputs_g, g_targets)
            
            g_loss.backward()
            g_optimizer.step()
            
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            
    print("\nTraining completed! Saving GAN models...")
    
    # Save the models
    os.makedirs("saved_gan_models", exist_ok=True)
    torch.save(generator.state_dict(), "saved_gan_models/generator.pt")
    torch.save(discriminator.state_dict(), "saved_gan_models/discriminator.pt")
    print("Saved generator.pt and discriminator.pt to 'saved_gan_models/'")
    
    # 6.5 Generate Synthetic Data
    print("\nGenerating 1,000 synthetic samples from the trained GAN...")
    synthetic_df = generate_synthetic_data(
        generator, 1000, scaler, encoder, 
        numerical_cols, categorical_cols, cat_dims, 
        latent_dim, device
    )
    
    # Save synthetic CSV
    synthetic_csv = "synthetic_crop_fertilizer.csv"
    synthetic_df.to_csv(synthetic_csv, index=False)
    print(f"Synthetic dataset saved successfully to '{synthetic_csv}'!")
    print(f"Sample of synthetic data generated:")
    print(synthetic_df.head())
    
    # 6.6 Visualization Comparison
    plot_distributions(df, synthetic_df, save_path="gan_comparison.png")

if __name__ == "__main__":
    main()
