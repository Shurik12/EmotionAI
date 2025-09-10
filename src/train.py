# Standard imports
import os
import csv
import sys
import pickle
import random
import pathlib
from random import shuffle
from itertools import chain
from tqdm import tqdm

# Data handling
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ML/Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix

# Custom imports
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "..", "emotiefflib", "backbones"))
import mobilefacenet
from robust_optimization import RobustOptimizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Torch version: {torch.__version__}")

# Configuration class to manage all settings
class Config:
    # Paths
    ALL_DATA_DIR = '/home/HDD6TB/datasets/emotions/'
    AFFECT_DATA_DIR = os.path.join(ALL_DATA_DIR, 'AffectNet/')
    IMG_AFFECT_DATA_DIR = os.path.join(AFFECT_DATA_DIR, 'Manually_Annotated_Images/')
    
    # File paths
    AFFECT_TRAIN_FILE = os.path.join(AFFECT_DATA_DIR, 'training.csv')
    AFFECT_TRAIN_FILTERED_FILE = os.path.join(AFFECT_DATA_DIR, 'training_filtered.csv')
    AFFECT_VAL_FILE = os.path.join(AFFECT_DATA_DIR, 'validation.csv')
    AFFECT_VAL_FILTERED_FILE = os.path.join(AFFECT_DATA_DIR, 'validation_filtered.csv')
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 40
    LEARNING_RATE = 3e-5
    GAMMA = 0.7
    SEED = 42
    IMG_SIZE = 224  # Default size
    
    # Emotion mappings
    AFFECTNET_EXPR2EMOTION = {
        0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 
        3: 'Surprise', 4: 'Fear', 5: 'Disgust', 
        6: 'Anger', 7: 'Contempt'
    }
    
    def __init__(self, use_enet2=False, seven_emotions=False, aligned=False, orig=False):
        self.USE_ENET2 = use_enet2
        self.SEVEN_EMOTIONS = seven_emotions
        self.ALIGNED = aligned
        self.ORIG = orig
        self.IMG_SIZE = 260 if use_enet2 else 224
        
        self._setup_paths()
        
    def _setup_paths(self):
        """Setup directory paths based on configuration"""
        size_str = str(self.IMG_SIZE)
        emotion_str = "seven_emotions" if self.SEVEN_EMOTIONS else "full_res"
        aligned_str = "_aligned" if self.ALIGNED else ""
        orig_str = "orig" if self.ORIG else size_str
        
        # Training directories
        if self.ORIG:
            self.TRAIN_DIR = os.path.join(self.AFFECT_DATA_DIR, 'orig', 'train')
            self.VAL_DIR = os.path.join(self.AFFECT_DATA_DIR, 'orig', 'val')
        elif self.ALIGNED:
            base = f"full_res_aligned{'/seven_emotions' if self.SEVEN_EMOTIONS else ''}"
            self.TRAIN_DIR = os.path.join(self.AFFECT_DATA_DIR, base, 'train')
            self.VAL_DIR = os.path.join(self.AFFECT_DATA_DIR, base, 'val')
        else:
            base = f"{size_str}{'/seven_emotions' if self.SEVEN_EMOTIONS else ''}"
            self.TRAIN_DIR = os.path.join(self.AFFECT_DATA_DIR, base, 'train')
            self.VAL_DIR = os.path.join(self.AFFECT_DATA_DIR, base, 'val')
        
        print(f"Train directory: {self.TRAIN_DIR}")
        print(f"Validation directory: {self.VAL_DIR}")

# Utility functions
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def save_csv(filename, outfile, dir_to_save, emotion_labels):
    """Filter and save CSV file with emotion data"""
    affect_df = pd.read_csv(filename)
    affect_vals = [d for _, d in affect_df.iterrows()]
    
    with open(os.path.join(dir_to_save, outfile), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filepath', 'emotion', 'valence', 'arousal'])
        writer.writeheader()
        
        for d in affect_vals:
            if d.expression >= len(emotion_labels) or d.face_width < 0:
                continue
                
            input_path = os.path.join(Config.IMG_AFFECT_DATA_DIR, d.subDirectory_filePath)
            dst_file_path = os.path.join(emotion_labels[d.expression], os.path.basename(d.subDirectory_filePath))
            
            if os.path.exists(os.path.join(dir_to_save, dst_file_path)):
                writer.writerow({
                    'filepath': dst_file_path,
                    'emotion': emotion_labels[d.expression],
                    'valence': d.valence, 
                    'arousal': d.arousal
                })

def ConcordanceCorCoeff(prediction, ground_truth):
    """Calculate Concordance Correlation Coefficient"""
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

def ConcordanceCorCoeffLoss(prediction, ground_truth):
    """CCC-based loss function"""
    return (1 - ConcordanceCorCoeff(prediction, ground_truth)) / 2

def label_smooth(target, n_classes, label_smoothing=0.1):
    """Apply label smoothing"""
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    
    # Label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target, weights=None):
    """Cross entropy with soft targets"""
    if weights is not None:
        return torch.mean(torch.sum(-weights * soft_target * F.log_softmax(pred, -1), 1))
    return torch.mean(torch.sum(-soft_target * F.log_softmax(pred, -1), 1))

def cross_entropy_with_label_smoothing(pred, target, weights=None, num_classes=None):
    """Cross entropy with label smoothing"""
    if num_classes is None:
        num_classes = pred.size(1)
    soft_target = label_smooth(target, num_classes)
    return cross_entropy_loss_with_soft_target(pred, soft_target, weights)

# Dataset classes
class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning (emotion + valence + arousal)"""
    def __init__(self, csv_file, root_dir, transform, class_to_idx):
        df = pd.read_csv(csv_file)
        df = df[df['emotion'].isin(class_to_idx.keys())]
        
        self.paths = list(df.filepath)
        self.targets = np.array([class_to_idx[cls] for cls in df.emotion])
        self.valence_arousal = df[['valence', 'arousal']].to_numpy()
        self.transform = transform
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.paths[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Get labels
        emotion_label = self.targets[idx]
        valence = torch.tensor(float(self.valence_arousal[idx, 0]), dtype=torch.float32)
        arousal = torch.tensor(float(self.valence_arousal[idx, 1]), dtype=torch.float32)
        
        return img, (emotion_label, valence, arousal)

# Model-related functions
def create_model(model_name, num_classes, pretrained_path=None):
    """Create and initialize model"""
    if model_name == 'tf_efficientnet_b0_ns':
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        model.classifier = torch.nn.Identity()
        
        if pretrained_path and os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            
        model.classifier = nn.Linear(in_features=1280, out_features=num_classes)
        
    elif model_name == 'mobilefacenet':
        model = mobilefacenet.MobileFaceNet()
        # Adjust output layer if needed
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)

def set_parameter_requires_grad(model, requires_grad):
    """Freeze or unfreeze model parameters"""
    for param in model.parameters():
        param.requires_grad = requires_grad

class MultiTaskLossWrapper(nn.Module):
    """Loss wrapper for multi-task learning"""
    def __init__(self, num_classes, class_weights=None):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_classes = num_classes
        
        if class_weights is not None:
            self.loss_emotions = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_emotions = nn.CrossEntropyLoss()
            
        self.loss_valence = ConcordanceCorCoeffLoss
        self.loss_arousal = ConcordanceCorCoeffLoss

    def forward(self, preds, target):
        loss_emotions = self.loss_emotions(preds[:, :self.num_classes], target[0])
        loss_valence = self.loss_valence(preds[:, self.num_classes], target[1])
        loss_arousal = self.loss_arousal(preds[:, self.num_classes + 1], target[2])
        
        return loss_emotions + (loss_valence + loss_arousal)

def train_model(model, train_loader, test_loader, criterion, n_epochs, learning_rate, 
                robust=False, num_classes=None, class_weights=None):
    """Train a model"""
    # Set up optimizer
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
    
    best_acc = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            data = data.to(device)
            
            if isinstance(labels, tuple):  # Multi-task
                labels = [label.to(device) for label in labels]
            else:  # Single task
                labels = labels.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            if isinstance(labels, list):  # Multi-task
                loss = criterion(output, labels)
            else:  # Single task
                if class_weights is not None:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, weights=class_weights, num_classes=num_classes
                    )
                else:
                    loss = F.cross_entropy(output, labels)
            
            # Backward pass
            if robust:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass
                output = model(data)
                if isinstance(labels, list):
                    loss = criterion(output, labels)
                else:
                    if class_weights is not None:
                        loss = cross_entropy_with_label_smoothing(
                            output, labels, weights=class_weights, num_classes=num_classes
                        )
                    else:
                        loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            if isinstance(labels, list):  # Multi-task
                acc = (output[:, :num_classes].argmax(dim=1) == labels[0]).float().mean()
            else:  # Single task
                acc = (output.argmax(dim=1) == labels).float().mean()
                
            epoch_accuracy += acc.item() * data.size(0)
            epoch_loss += loss.item() * data.size(0)
        
        # Calculate epoch metrics
        epoch_accuracy /= len(train_loader.dataset)
        epoch_loss /= len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                
                if isinstance(labels, tuple):  # Multi-task
                    labels = [l.to(device) for l in labels]
                else:  # Single task
                    labels = labels.to(device)
                
                output = model(data)
                
                if isinstance(labels, list):  # Multi-task
                    loss = criterion(output, labels)
                    acc = (output[:, :num_classes].argmax(dim=1) == labels[0]).float().mean()
                else:  # Single task
                    if class_weights is not None:
                        loss = cross_entropy_with_label_smoothing(
                            output, labels, weights=class_weights, num_classes=num_classes
                        )
                    else:
                        loss = F.cross_entropy(output, labels)
                    acc = (output.argmax(dim=1) == labels).float().mean()
                
                val_accuracy += acc.item() * data.size(0)
                val_loss += loss.item() * data.size(0)
        
        val_accuracy /= len(test_loader.dataset)
        val_loss /= len(test_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_acc:.4f}")
    
    return model, history

def evaluate_model(model, test_loader, class_to_idx, transforms, device):
    """Evaluate model performance"""
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for class_name in tqdm(os.listdir(test_loader.dataset.root)):
            if class_name in class_to_idx:
                class_dir = os.path.join(test_loader.dataset.root, class_name)
                true_label = class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transforms(img).unsqueeze(0).to(device)
                    
                    output = model(img_tensor)
                    scores = output[0].cpu().numpy()
                    pred_label = np.argmax(scores)
                    
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    y_scores.append(scores)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    accuracy = 100.0 * (y_true == y_pred).sum() / len(y_true)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    for i, class_name in class_to_idx.items():
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = 100.0 * (y_pred[class_mask] == i).sum() / class_mask.sum()
            print(f"{class_name}: {class_acc:.2f}% ({class_mask.sum()} samples)")
    
    return y_true, y_pred, y_scores

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set up configuration
    config = Config(use_enet2=False, seven_emotions=False)
    set_seed(config.SEED)
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate class weights
    unique, counts = np.unique(train_dataset.targets, return_counts=True)
    class_weights = torch.FloatTensor(1 / counts).to(device)
    class_weights = class_weights / class_weights.min()
    
    # Get class mappings
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # Create and train model
    model = create_model(
        'tf_efficientnet_b0_ns', 
        num_classes, 
        pretrained_path='../../models/pretrained_faces/state_vggface2_enet0_new.pt'
    )
    
    # Train the model
    trained_model, history = train_model(
        model, train_loader, test_loader, 
        criterion=None,  # Will use default in training function
        n_epochs=6, 
        learning_rate=1e-4, 
        robust=True,
        num_classes=num_classes,
        class_weights=class_weights
    )
    
    # Save the model
    model_path = '../../models/affectnet_emotions/enet_b0_8_best.pt'
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the model
    y_true, y_pred, y_scores = evaluate_model(
        trained_model, test_loader, class_to_idx, test_transforms, device
    )
    
    # Plot confusion matrix
    class_names = list(class_to_idx.keys())
    plot_confusion_matrix(y_true, y_pred, class_names)