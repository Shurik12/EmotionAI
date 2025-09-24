import os
import timm
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from training.robust_optimization import RobustOptimizer
import copy
from typing import Dict, List, Tuple, Optional, Union, Any

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning (emotion + valence + arousal)."""
    
    def __init__(self, file_path: str, transform: transforms.Compose, class_to_idx: Dict[str, int]):
        df = pd.read_csv(file_path, sep="\t")
        df = df[df['emotion'].isin(class_to_idx.keys())]
        
        self.dir = "processed_images/"
        self.paths = list(df["file"])
        self.targets = np.array([class_to_idx[cls] for cls in df["emotion"]])
        self.valence_arousal = df[["valence", "arousal"]].to_numpy()
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Load image
        img_path = self.dir + self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Get labels
        emotion_label = torch.tensor(self.targets[idx], dtype=torch.long)
        valence = torch.tensor(float(self.valence_arousal[idx, 0]), dtype=torch.float32)
        arousal = torch.tensor(float(self.valence_arousal[idx, 1]), dtype=torch.float32)
        
        return img, (emotion_label, valence, arousal)

class MultiTaskLossWrapper(nn.Module):
    """Multi-task loss wrapper combining emotion classification and valence/arousal regression."""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.loss_emotions = nn.CrossEntropyLoss(weight=class_weights)
        self.loss_valence = self.concordance_correlation_coefficient_loss
        self.loss_arousal = self.concordance_correlation_coefficient_loss

    def concordance_correlation_coefficient(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Calculate Concordance Correlation Coefficient."""
        mean_gt = torch.mean(ground_truth, 0)
        mean_pred = torch.mean(prediction, 0)
        var_gt = torch.var(ground_truth, 0)
        var_pred = torch.var(prediction, 0)
        
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        
        cor = torch.sum(v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
        sd_gt = torch.std(ground_truth)
        sd_pred = torch.std(prediction)
        
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        
        return numerator / denominator

    def concordance_correlation_coefficient_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """CCC-based loss function."""
        return (1 - self.concordance_correlation_coefficient(prediction, ground_truth)) / 2

    def forward(self, preds: torch.Tensor, target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss_emotions = self.loss_emotions(preds[:, :self.num_classes], target[0])
        loss_valence = self.loss_valence(preds[:, self.num_classes], target[1])
        loss_arousal = self.loss_arousal(preds[:, self.num_classes + 1], target[2])
        return loss_emotions + (loss_valence + loss_arousal)

class MultiTaskModel(nn.Module):
    """Multi-task model wrapper."""
    
    def __init__(self, base_model: nn.Module, num_classes: int, feature_dim: int = 1280):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(in_features=feature_dim, out_features=num_classes + 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)

def set_parameter_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Set requires_grad for all parameters in model."""
    for param in model.parameters():
        param.requires_grad = requires_grad

def create_data_loaders(
    train_file: str,
    test_file: str,
    class_to_idx: Dict[str, int],
    img_size: int = 260,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dict[int, float]]:
    """Create train and test data loaders."""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = MultiTaskDataset(train_file, transform, class_to_idx)
    test_dataset = MultiTaskDataset(test_file, transform, class_to_idx)

    # Calculate class weights
    unique, counts = np.unique(train_dataset.targets, return_counts=True)
    class_weights = {i: 1.0 / count for i, count in zip(unique, counts)}
    
    # Data loaders
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Class counts: {dict(zip(unique, counts))}")
    print(f"Class weights: {class_weights}")

    return train_loader, test_loader, class_weights

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int,
    robust: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_mse_valence = 0.0
    epoch_mse_arousal = 0.0
    
    for data, labels in tqdm(data_loader, desc="Training"):
        data = data.to(device)
        labels = [label.to(device) for label in labels]
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass
        if robust and isinstance(optimizer, RobustOptimizer):
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass for robust optimization
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate metrics
        acc = (outputs[:, :num_classes].argmax(dim=1) == labels[0]).float().sum()
        epoch_accuracy += acc.item()
        
        mse_valence = ((outputs[:, num_classes] - labels[1])**2).float().sum()
        epoch_mse_valence += mse_valence.item()
        
        mse_arousal = ((outputs[:, num_classes + 1] - labels[2])**2).float().sum()
        epoch_mse_arousal += mse_arousal.item()
        
        epoch_loss += loss.item()

    # Normalize metrics
    num_samples = len(data_loader.dataset)
    metrics = {
        'loss': epoch_loss / num_samples,
        'accuracy': epoch_accuracy / num_samples,
        'mse_valence': epoch_mse_valence / num_samples,
        'mse_arousal': epoch_mse_arousal / num_samples
    }
    
    return metrics

def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_mse_valence = 0.0
    val_mse_arousal = 0.0
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Validation"):
            data = data.to(device)
            labels = [label.to(device) for label in labels]
            
            outputs = model(data)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_accuracy += (outputs[:, :num_classes].argmax(dim=1) == labels[0]).float().sum().item()
            val_mse_valence += ((outputs[:, num_classes] - labels[1])**2).float().sum().item()
            val_mse_arousal += ((outputs[:, num_classes + 1] - labels[2])**2).float().sum().item()

    num_samples = len(data_loader.dataset)
    metrics = {
        'loss': val_loss / num_samples,
        'accuracy': val_accuracy / num_samples,
        'mse_valence': val_mse_valence / num_samples,
        'mse_arousal': val_mse_arousal / num_samples
    }
    
    return metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    n_epochs: int = 40,
    learning_rate: float = 3e-5,
    robust: bool = False
) -> nn.Module:
    """Train the multi-task model."""
    # Optimizer
    if robust:
        optimizer = RobustOptimizer(
            filter(lambda p: p.requires_grad, model.parameters()),
            optim.Adam,
            lr=learning_rate
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, num_classes, robust)
        
        # Validate
        val_metrics = validate(model, test_loader, criterion, device, num_classes)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"MSE Valence: {train_metrics['mse_valence']:.4f}, MSE Arousal: {train_metrics['mse_arousal']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"MSE Valence: {val_metrics['mse_valence']:.4f}, MSE Arousal: {val_metrics['mse_arousal']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model with accuracy: {best_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with accuracy: {best_acc:.4f}")
    else:
        print("No best model found")
    
    return model

def main():
    """Main function to start the training process."""
    # Configuration
    batch_size = 64
    epochs = 40
    learning_rate = 3e-5
    gamma = 0.7
    seed = 42
    img_size = 260
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Class mappings
    idx_to_class = {
        0: 'Anger', 
        1: 'Contempt', 
        2: 'Disgust', 
        3: 'Fear', 
        4: 'Happiness', 
        5: 'Neutral', 
        6: 'Sadness', 
        7: 'Surprise'
    }
    class_to_idx = {cls: idx for idx, cls in idx_to_class.items()}
    num_classes = len(class_to_idx)
    
    # Create data loaders
    train_loader, test_loader, class_weights = create_data_loaders(
        "train.tsv", "test.tsv", class_to_idx, img_size, batch_size
    )

    print(f"Number of classes: {num_classes}")

    # Load pre-trained model
    base_model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    base_model.classifier = torch.nn.Identity()
    
    # Load pre-trained weights
    pretrained_path = 'pretrained_model.pt'
    if os.path.exists(pretrained_path):
        try:
            # First try to load as state dict
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict):
                base_model.load_state_dict(state_dict)
                print("Loaded pre-trained weights successfully")
            else:
                # If it's a full model, extract its state dict
                base_model.load_state_dict(state_dict.state_dict())
                print("Loaded pre-trained model and extracted state dict")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Training from scratch")
    else:
        print("Pre-trained weights not found, training from scratch")
    
    # Create multi-task model
    model = MultiTaskModel(base_model, num_classes)
    model = model.to(device)
    print(model)

    # Create loss function with class weights
    weights_tensor = torch.FloatTensor(list(class_weights.values())).to(device)
    criterion = MultiTaskLossWrapper(weights_tensor, num_classes)

    # Freeze base model parameters
    set_parameter_requires_grad(model.base_model, requires_grad=False)
    set_parameter_requires_grad(model.classifier, requires_grad=True)

    # Train the model
    train_model(
        model, train_loader, test_loader, criterion, device, num_classes,
        n_epochs=3, learning_rate=0.001, robust=True
    )

    torch.save(model, "model_v0.0.1.pt")

if __name__ == "__main__":
    main()