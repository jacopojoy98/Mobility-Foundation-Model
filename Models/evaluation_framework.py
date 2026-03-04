"""
Evaluation Framework for Mobility Foundation Model
==================================================

This module provides downstream tasks to evaluate the quality of learned embeddings:
1. Destination Prediction
2. Time of Arrival Prediction
3. Next Location Prediction
4. Trip Purpose Classification
5. User Identification
6. Trajectory Similarity Ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Tuple
import json


class DownstreamTaskDataset(Dataset):
    """
    Generic dataset for downstream evaluation tasks.
    """
    def __init__(self, embeddings, labels, task_type='classification'):
        """
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of labels
            task_type: 'classification' or 'regression'
        """
        self.embeddings = embeddings
        self.labels = labels
        self.task_type = task_type
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class LinearProbe(nn.Module):
    """
    Simple linear classifier/regressor for probing embeddings.
    """
    def __init__(self, input_dim, output_dim, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    """
    MLP classifier/regressor for probing embeddings.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


def train_probe(
    probe_model,
    train_loader,
    val_loader,
    task_type='classification',
    num_epochs=500,
    learning_rate=1e-3,
    device='cuda'
):
    """
    Train a probe model on frozen embeddings.
    
    Args:
        probe_model: Linear or MLP probe
        train_loader: Training data loader
        val_loader: Validation data loader
        task_type: 'classification' or 'regression'
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        best_model: Best model state dict
        train_metrics: Training metrics
        val_metrics: Validation metrics
    """
    probe_model = probe_model.to(device)
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=learning_rate)
    
    best_val_metric = float('inf') if task_type == 'regression' else 0.0
    best_model_state = None
    
    train_metrics = {'loss': [], 'metric': []}
    val_metrics = {'loss': [], 'metric': []}
    
    for epoch in range(num_epochs):
        # Training
        probe_model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe_model(embeddings)
            
            if task_type == 'classification':
                loss = F.cross_entropy(outputs, labels)
                preds = outputs.argmax(dim=1)
            else:  # regression
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                loss = F.mse_loss(outputs, labels)
                preds = outputs
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(preds.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())
        
        # Validation
        probe_model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = probe_model(embeddings)
                
                if task_type == 'classification':
                    loss = F.cross_entropy(outputs, labels)
                    preds = outputs.argmax(dim=1)
                else:  # regression
                    if outputs.dim() > 1 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(1)
                    loss = F.mse_loss(outputs, labels)
                    preds = outputs
                
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if task_type == 'classification':
            train_metric = accuracy_score(train_labels, train_preds)
            val_metric = accuracy_score(val_labels, val_preds)
        else:  # regression
            train_metric = mean_absolute_error(train_labels, train_preds)
            val_metric = mean_absolute_error(val_labels, val_preds)
        
        train_metrics['loss'].append(train_loss)
        train_metrics['metric'].append(train_metric)
        val_metrics['loss'].append(val_loss)
        val_metrics['metric'].append(val_metric)
        
        # Save best model
        if task_type == 'classification':
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = probe_model.state_dict().copy()
        else:  # regression (lower is better)
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_model_state = probe_model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            metric_name = 'Accuracy' if task_type == 'classification' else 'MAE'
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}")
    
    return best_model_state, train_metrics, val_metrics


class MobilityEvaluationSuite:
    """
    Complete evaluation suite for mobility foundation models.
    """
    def __init__(self, foundation_model, device='cuda'):
        """
        Args:
            foundation_model: The trained mobility foundation model
            device: Device to run evaluations on
        """
        self.foundation_model = foundation_model
        self.device = device
        self.foundation_model.eval()
        self.foundation_model.to(device)
    
    def extract_embeddings(self, sequences, sequence_type='trajectory', padding_mask=None):
        """
        Extract embeddings from the foundation model.
        
        Args:
            sequences: Input sequences (batch_size, seq_len, token_dim)
            sequence_type: 'trajectory' or 'visit'
            padding_mask: Optional padding mask
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        with torch.no_grad():
            sequences = sequences.to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            
            if sequence_type == 'trajectory':
                embeddings = self.foundation_model.encode_trajectory(sequences, padding_mask)
            else:  # visit
                embeddings = self.foundation_model.encode_visits(sequences, padding_mask)
        
        return embeddings.cpu()
    
    def evaluate_destination_prediction(
        self,
        train_trajectories,
        train_destinations,
        val_trajectories,
        val_destinations,
        num_destinations,
        sequence_type='trajectory',
        probe_type='mlp'
    ):
        """
        Evaluate destination prediction task.
        
        Args:
            train_trajectories: Training trajectory sequences
            train_destinations: Training destination labels (class indices)
            val_trajectories: Validation trajectories
            val_destinations: Validation destinations
            num_destinations: Number of unique destinations
            sequence_type: 'trajectory' or 'visit'
            probe_type: 'linear' or 'mlp'
            
        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*70)
        print("TASK 1: DESTINATION PREDICTION")
        print("="*70)
        print(f"Number of destination classes: {num_destinations}")
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings = self.extract_embeddings(train_trajectories, sequence_type)
        val_embeddings = self.extract_embeddings(val_trajectories, sequence_type)
        
        # Create datasets
        train_dataset = DownstreamTaskDataset(train_embeddings, train_destinations, 'classification')
        val_dataset = DownstreamTaskDataset(val_embeddings, val_destinations, 'classification')
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create probe
        embedding_dim = train_embeddings.size(1)
        if probe_type == 'linear':
            probe = LinearProbe(embedding_dim, num_destinations, 'classification')
        else:
            probe = MLPProbe(embedding_dim, num_destinations, hidden_dim=256, task_type='classification')
        
        print(f"Training {probe_type} probe...")
        best_model, train_metrics, val_metrics = train_probe(
            probe, train_loader, val_loader, 
            task_type='classification',
            num_epochs=500,
            device=self.device
        )
        
        # Final evaluation
        probe.load_state_dict(best_model)
        probe.eval()
        
        with torch.no_grad():
            val_preds = []
            val_labels_list = []
            for embeddings, labels in val_loader:
                outputs = probe(embeddings.to(self.device))
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        accuracy = accuracy_score(val_labels_list, val_preds)
        f1 = f1_score(val_labels_list, val_preds, average='weighted')
        
        results = {
            'task': 'destination_prediction',
            'accuracy': accuracy,
            'f1_score': f1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return results
    
    def evaluate_time_of_arrival(
        self,
        train_trajectories,
        train_arrival_times,
        val_trajectories,
        val_arrival_times,
        sequence_type='trajectory',
        probe_type='mlp'
    ):
        """
        Evaluate time of arrival prediction task.
        
        Args:
            train_trajectories: Training trajectory sequences
            train_arrival_times: Training arrival times (in minutes or normalized)
            val_trajectories: Validation trajectories
            val_arrival_times: Validation arrival times
            sequence_type: 'trajectory' or 'visit'
            probe_type: 'linear' or 'mlp'
            
        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*70)
        print("TASK 2: TIME OF ARRIVAL PREDICTION")
        print("="*70)
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings = self.extract_embeddings(train_trajectories, sequence_type)
        val_embeddings = self.extract_embeddings(val_trajectories, sequence_type)
        
        # Create datasets
        train_dataset = DownstreamTaskDataset(
            train_embeddings, 
            torch.FloatTensor(train_arrival_times), 
            'regression'
        )
        val_dataset = DownstreamTaskDataset(
            val_embeddings, 
            torch.FloatTensor(val_arrival_times), 
            'regression'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create probe
        embedding_dim = train_embeddings.size(1)
        if probe_type == 'linear':
            probe = LinearProbe(embedding_dim, 1, 'regression')
        else:
            probe = MLPProbe(embedding_dim, 1, hidden_dim=256, task_type='regression')
        
        print(f"Training {probe_type} probe...")
        best_model, train_metrics, val_metrics = train_probe(
            probe, train_loader, val_loader, 
            task_type='regression',
            num_epochs=500,
            device=self.device
        )
        
        # Final evaluation
        probe.load_state_dict(best_model)
        probe.eval()
        
        with torch.no_grad():
            val_preds = []
            val_labels_list = []
            for embeddings, labels in val_loader:
                outputs = probe(embeddings.to(self.device))
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(1)
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        mae = mean_absolute_error(val_labels_list, val_preds)
        rmse = np.sqrt(mean_squared_error(val_labels_list, val_preds))
        
        results = {
            'task': 'time_of_arrival',
            'mae': mae,
            'rmse': rmse,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"\nResults:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        return results
    
    def evaluate_next_location(
        self,
        train_trajectories,
        train_next_locations,
        val_trajectories,
        val_next_locations,
        num_locations,
        sequence_type='trajectory',
        probe_type='mlp'
    ):
        """
        Evaluate next location prediction task.
        
        Args:
            train_trajectories: Training trajectory sequences
            train_next_locations: Training next location labels
            val_trajectories: Validation trajectories
            val_next_locations: Validation next locations
            num_locations: Number of unique locations
            sequence_type: 'trajectory' or 'visit'
            probe_type: 'linear' or 'mlp'
            
        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*70)
        print("TASK 3: NEXT LOCATION PREDICTION")
        print("="*70)
        print(f"Number of location classes: {num_locations}")
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings = self.extract_embeddings(train_trajectories, sequence_type)
        val_embeddings = self.extract_embeddings(val_trajectories, sequence_type)
        
        # Create datasets
        train_dataset = DownstreamTaskDataset(train_embeddings, train_next_locations, 'classification')
        val_dataset = DownstreamTaskDataset(val_embeddings, val_next_locations, 'classification')
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create probe
        embedding_dim = train_embeddings.size(1)
        if probe_type == 'linear':
            probe = LinearProbe(embedding_dim, num_locations, 'classification')
        else:
            probe = MLPProbe(embedding_dim, num_locations, hidden_dim=256, task_type='classification')
        
        print(f"Training {probe_type} probe...")
        best_model, train_metrics, val_metrics = train_probe(
            probe, train_loader, val_loader, 
            task_type='classification',
            num_epochs=500,
            device=self.device
        )
        
        # Final evaluation with top-k accuracy
        probe.load_state_dict(best_model)
        probe.eval()
        
        with torch.no_grad():
            val_preds = []
            val_top5 = []
            val_labels_list = []
            for embeddings, labels in val_loader:
                outputs = probe(embeddings.to(self.device))
                preds = outputs.argmax(dim=1)
                top5 = outputs.topk(5, dim=1).indices
                val_preds.extend(preds.cpu().numpy())
                val_top5.append(top5.cpu())
                val_labels_list.extend(labels.numpy())
        
        accuracy = accuracy_score(val_labels_list, val_preds)
        
        # Top-5 accuracy
        val_top5 = torch.cat(val_top5, dim=0)
        val_labels_tensor = torch.LongTensor(val_labels_list).unsqueeze(1)
        top5_acc = (val_top5 == val_labels_tensor).any(dim=1).float().mean().item()
        
        results = {
            'task': 'next_location',
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"\nResults:")
        print(f"  Top-1 Accuracy: {accuracy:.4f}")
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")
        
        return results
    
    def evaluate_trip_purpose(
        self,
        train_trajectories,
        train_purposes,
        val_trajectories,
        val_purposes,
        num_purposes,
        sequence_type='trajectory',
        probe_type='mlp'
    ):
        """
        Evaluate trip purpose classification task.
        
        Args:
            train_trajectories: Training trajectory sequences
            train_purposes: Training purpose labels (e.g., 0=work, 1=home, 2=leisure)
            val_trajectories: Validation trajectories
            val_purposes: Validation purposes
            num_purposes: Number of purpose categories
            sequence_type: 'trajectory' or 'visit'
            probe_type: 'linear' or 'mlp'
            
        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*70)
        print("TASK 4: TRIP PURPOSE CLASSIFICATION")
        print("="*70)
        print(f"Number of purpose classes: {num_purposes}")
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings = self.extract_embeddings(train_trajectories, sequence_type)
        val_embeddings = self.extract_embeddings(val_trajectories, sequence_type)
        
        # Create datasets
        train_dataset = DownstreamTaskDataset(train_embeddings, train_purposes, 'classification')
        val_dataset = DownstreamTaskDataset(val_embeddings, val_purposes, 'classification')
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create probe
        embedding_dim = train_embeddings.size(1)
        if probe_type == 'linear':
            probe = LinearProbe(embedding_dim, num_purposes, 'classification')
        else:
            probe = MLPProbe(embedding_dim, num_purposes, hidden_dim=256, task_type='classification')
        
        print(f"Training {probe_type} probe...")
        best_model, train_metrics, val_metrics = train_probe(
            probe, train_loader, val_loader, 
            task_type='classification',
            num_epochs=500,
            device=self.device
        )
        
        # Final evaluation
        probe.load_state_dict(best_model)
        probe.eval()
        
        with torch.no_grad():
            val_preds = []
            val_labels_list = []
            for embeddings, labels in val_loader:
                outputs = probe(embeddings.to(self.device))
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        accuracy = accuracy_score(val_labels_list, val_preds)
        f1 = f1_score(val_labels_list, val_preds, average='weighted')
        
        results = {
            'task': 'trip_purpose',
            'accuracy': accuracy,
            'f1_score': f1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return results
    
    def evaluate_user_identification(
        self,
        train_trajectories,
        train_user_ids,
        val_trajectories,
        val_user_ids,
        num_users,
        sequence_type='trajectory',
        probe_type='mlp'
    ):
        """
        Evaluate user identification task.
        Tests if embeddings capture user-specific mobility patterns.
        
        Args:
            train_trajectories: Training trajectory sequences
            train_user_ids: Training user IDs
            val_trajectories: Validation trajectories
            val_user_ids: Validation user IDs
            num_users: Number of unique users
            sequence_type: 'trajectory' or 'visit'
            probe_type: 'linear' or 'mlp'
            
        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*70)
        print("TASK 5: USER IDENTIFICATION")
        print("="*70)
        print(f"Number of users: {num_users}")
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings = self.extract_embeddings(train_trajectories, sequence_type)
        val_embeddings = self.extract_embeddings(val_trajectories, sequence_type)
        
        # Create datasets
        train_dataset = DownstreamTaskDataset(train_embeddings, train_user_ids, 'classification')
        val_dataset = DownstreamTaskDataset(val_embeddings, val_user_ids, 'classification')
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Create probe
        embedding_dim = train_embeddings.size(1)
        if probe_type == 'linear':
            probe = LinearProbe(embedding_dim, num_users, 'classification')
        else:
            probe = MLPProbe(embedding_dim, num_users, hidden_dim=256, task_type='classification')
        
        print(f"Training {probe_type} probe...")
        best_model, train_metrics, val_metrics = train_probe(
            probe, train_loader, val_loader, 
            task_type='classification',
            num_epochs=500,
            device=self.device
        )
        
        # Final evaluation
        probe.load_state_dict(best_model)
        probe.eval()
        
        with torch.no_grad():
            val_preds = []
            val_labels_list = []
            for embeddings, labels in val_loader:
                outputs = probe(embeddings.to(self.device))
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        accuracy = accuracy_score(val_labels_list, val_preds)
        f1 = f1_score(val_labels_list, val_preds, average='weighted')
        
        results = {
            'task': 'user_identification',
            'accuracy': accuracy,
            'f1_score': f1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return results
    
    def run_full_evaluation(self, evaluation_data, save_path='evaluation_results.json'):
        """
        Run all evaluation tasks and save results.
        
        Args:
            evaluation_data: Dictionary containing data for all tasks
            save_path: Path to save results
            
        Returns:
            all_results: Dictionary with all task results
        """
        all_results = {}
        
        # Task 1: Destination Prediction
        if 'destination' in evaluation_data:
            results = self.evaluate_destination_prediction(**evaluation_data['destination'])
            all_results['destination_prediction'] = results
        
        # Task 2: Time of Arrival
        if 'time_of_arrival' in evaluation_data:
            results = self.evaluate_time_of_arrival(**evaluation_data['time_of_arrival'])
            all_results['time_of_arrival'] = results
        
        # Task 3: Next Location
        if 'next_location' in evaluation_data:
            results = self.evaluate_next_location(**evaluation_data['next_location'])
            all_results['next_location'] = results
        
        # Task 4: Trip Purpose
        if 'trip_purpose' in evaluation_data:
            results = self.evaluate_trip_purpose(**evaluation_data['trip_purpose'])
            all_results['trip_purpose'] = results
        
        # Task 5: User Identification
        if 'user_identification' in evaluation_data:
            results = self.evaluate_user_identification(**evaluation_data['user_identification'])
            all_results['user_identification'] = results
        
        # Save results
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {save_path}")
        
        return all_results


# Example usage
if __name__ == "__main__":
    print("Mobility Foundation Model Evaluation Framework")
    print("="*70)
    print("\nThis module provides comprehensive evaluation tasks:")
    print("  1. Destination Prediction - Predict final destination")
    print("  2. Time of Arrival - Predict arrival time")
    print("  3. Next Location - Predict next location in sequence")
    print("  4. Trip Purpose - Classify trip purpose (work/home/leisure)")
    print("  5. User Identification - Identify user from trajectory")
    print("\nSee evaluation_example.py for complete usage example.")



## Aggiungere l'errore di quanto sono "lontano", fare una k-accuracy, trajectory-reconstruction