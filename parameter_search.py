import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import product

from config import Config
from model import TransformerModel
from utils import (
    set_seed, process_data, split_data, 
    ShockDeathDataset, evaluate_model, save_results
)

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, config, device, look_back, shock_interval, death_interval):
    """Train the model"""
    best_val_loss = float('inf')
    best_death_auroc = 0.0  # Track best death prediction AUC
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"\nCurrent parameter combination: look_back={look_back}, shock_interval={shock_interval}, death_interval={death_interval}")
    
    # Add training progress bar
    pbar = tqdm(range(config.num_epochs), desc='Training progress')
    for epoch in pbar:
        # Training phase
        model.train()
        running_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', leave=False)
        for data in batch_pbar:
            inputs, labels = [d.to(device) for d in data]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['overall']['loss']
        current_death_auroc = val_metrics['class_1']['auroc']  # Current death prediction AUC
        
        # Test phase - Add test set evaluation
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        test_loss = test_metrics['overall']['loss']
        test_death_auroc = test_metrics['class_1']['auroc']
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Calculate and print current epoch's detailed metrics
        train_loss = running_loss / len(train_loader)
        
        # Print metrics for each class
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Test loss: {test_loss:.4f}")
        print(f"Death prediction validation AUC: {current_death_auroc:.4f}, Test AUC: {test_death_auroc:.4f}")
        
        print("Shock prediction metrics:")
        print(f"   Validation set - Accuracy: {val_metrics['class_0']['accuracy']:.4f}, Precision: {val_metrics['class_0']['precision']:.4f}, Recall: {val_metrics['class_0']['recall']:.4f}, F1: {val_metrics['class_0']['f1']:.4f}, AUROC: {val_metrics['class_0']['auroc']:.4f}")
        print(f"   Test set - Accuracy: {test_metrics['class_0']['accuracy']:.4f}, Precision: {test_metrics['class_0']['precision']:.4f}, Recall: {test_metrics['class_0']['recall']:.4f}, F1: {test_metrics['class_0']['f1']:.4f}, AUROC: {test_metrics['class_0']['auroc']:.4f}")
        print("   Validation set confusion matrix:")
        print(f"     True Negative (TN): {val_metrics['class_0']['confusion_matrix'][0][0]}, False Positive (FP): {val_metrics['class_0']['confusion_matrix'][0][1]}")
        print(f"     False Negative (FN): {val_metrics['class_0']['confusion_matrix'][1][0]}, True Positive (TP): {val_metrics['class_0']['confusion_matrix'][1][1]}")
        print("   Test set confusion matrix:")
        print(f"     True Negative (TN): {test_metrics['class_0']['confusion_matrix'][0][0]}, False Positive (FP): {test_metrics['class_0']['confusion_matrix'][0][1]}")
        print(f"     False Negative (FN): {test_metrics['class_0']['confusion_matrix'][1][0]}, True Positive (TP): {test_metrics['class_0']['confusion_matrix'][1][1]}")
        
        print("Death prediction metrics:")
        print(f"   Validation set - Accuracy: {val_metrics['class_1']['accuracy']:.4f}, Precision: {val_metrics['class_1']['precision']:.4f}, Recall: {val_metrics['class_1']['recall']:.4f}, F1: {val_metrics['class_1']['f1']:.4f}, AUROC: {val_metrics['class_1']['auroc']:.4f}")
        print(f"   Test set - Accuracy: {test_metrics['class_1']['accuracy']:.4f}, Precision: {test_metrics['class_1']['precision']:.4f}, Recall: {test_metrics['class_1']['recall']:.4f}, F1: {test_metrics['class_1']['f1']:.4f}, AUROC: {test_metrics['class_1']['auroc']:.4f}")
        print("   Validation set confusion matrix:")
        print(f"     True Negative (TN): {val_metrics['class_1']['confusion_matrix'][0][0]}, False Positive (FP): {val_metrics['class_1']['confusion_matrix'][0][1]}")
        print(f"     False Negative (FN): {val_metrics['class_1']['confusion_matrix'][1][0]}, True Positive (TP): {val_metrics['class_1']['confusion_matrix'][1][1]}")
        print("   Test set confusion matrix:")
        print(f"     True Negative (TN): {test_metrics['class_1']['confusion_matrix'][0][0]}, False Positive (FP): {test_metrics['class_1']['confusion_matrix'][0][1]}")
        print(f"     False Negative (FN): {test_metrics['class_1']['confusion_matrix'][1][0]}, True Positive (TP): {test_metrics['class_1']['confusion_matrix'][1][1]}")
        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update epoch progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_death_auc': f'{current_death_auroc:.4f}',
            'test_death_auc': f'{test_death_auroc:.4f}'
        })
        
        # Save model: Save whenever death AUC improves
        if current_death_auroc > best_death_auroc:
            improvement_message = f"【Death AUC improvement】from {best_death_auroc:.4f} to {current_death_auroc:.4f}"
            best_death_auroc = current_death_auroc
            best_model_state = model.state_dict()
            
            # Create save directory
            model_save_dir = os.path.join(config.checkpoint_dir, f"lb{look_back}_si{shock_interval}_di{death_interval}")
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
                
            # Save model
            model_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"{improvement_message}, saved best model to: {model_path}")
            print(f"Corresponding test set death AUC: {test_death_auroc:.4f}")
            
            # If validation loss also improves, reset early stopping counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"【Also】validation loss improved to: {val_loss:.4f}")
            else:
                print(f"But validation loss did not improve, current: {val_loss:.4f}, best: {best_val_loss:.4f}")
        else:
            # Only count early stopping when both death AUC and validation loss do not improve
            if val_loss >= best_val_loss:
                epochs_no_improve += 1
                print(f"Death AUC did not improve, validation loss did not improve. Early stopping count: {epochs_no_improve}/{config.early_stopping}")
                if epochs_no_improve >= config.early_stopping:
                    print(f"\nEarly stopping triggered! No improvement for {config.early_stopping} epochs")
                    break
            else:
                # If only validation loss improves, record it but don't reset early stopping counter
                best_val_loss = val_loss
                print(f"【Only validation loss improved】to {val_loss:.4f}, but death AUC did not improve, continuing training...")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Evaluate best model on test set
        best_test_metrics = evaluate_model(model, test_loader, criterion, device)
        best_test_death_auroc = best_test_metrics['class_1']['auroc']
        print(f"\nLoaded best model, validation set death prediction AUC: {best_death_auroc:.4f}, test set death prediction AUC: {best_test_death_auroc:.4f}")
    
    return model

def run_experiment(config, raw_data, raw_label, look_back, shock_interval, death_interval):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"Starting experiment: look_back={look_back}, shock_interval={shock_interval}, death_interval={death_interval}")
    print("="*80)
    
    # Ensure checkpoint directory exists
    experiment_dir = os.path.join(config.checkpoint_dir, f"lb{look_back}_si{shock_interval}_di{death_interval}")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    print(f"Models will be saved to: {experiment_dir}")
    
    print("\nProcessing data...")
    # Process data
    data, label = process_data(raw_data, raw_label, config, look_back, shock_interval, death_interval)
    train_label, val_label, test_label = split_data(label)
    
    print("Creating datasets...")
    # Create datasets
    delete_columns = ['stay_id', 'charttime', 'antibiotic_time', 'sepsis_time']
    train_dataset = ShockDeathDataset(train_label, data, delete_columns)
    val_dataset = ShockDeathDataset(val_label, data, delete_columns)
    test_dataset = ShockDeathDataset(test_label, data, delete_columns)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    W = len([col for col in data.columns if col not in delete_columns])
    model_config = config.get_model_config(look_back, W)
    model = TransformerModel(model_config).to(device)
    
    # Set training parameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=config.lr_factor, patience=config.lr_patience
    )
    
    print("\nStarting training...")
    # Train model
    model = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, config, device, look_back, shock_interval, death_interval)
    
    print("\nEvaluating model performance...")
    # Evaluate model
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    val_metrics = evaluate_model(model, val_loader, criterion, device)
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\nFinal evaluation results:")
    print("Shock prediction metrics:")
    print(f"   Training set - Accuracy: {train_metrics['class_0']['accuracy']:.4f}, Precision: {train_metrics['class_0']['precision']:.4f}, Recall: {train_metrics['class_0']['recall']:.4f}, F1: {train_metrics['class_0']['f1']:.4f}, AUROC: {train_metrics['class_0']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {train_metrics['class_0']['confusion_matrix'][0][0]}, False Positive (FP): {train_metrics['class_0']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {train_metrics['class_0']['confusion_matrix'][1][0]}, True Positive (TP): {train_metrics['class_0']['confusion_matrix'][1][1]}")
    
    print(f"   Validation set - Accuracy: {val_metrics['class_0']['accuracy']:.4f}, Precision: {val_metrics['class_0']['precision']:.4f}, Recall: {val_metrics['class_0']['recall']:.4f}, F1: {val_metrics['class_0']['f1']:.4f}, AUROC: {val_metrics['class_0']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {val_metrics['class_0']['confusion_matrix'][0][0]}, False Positive (FP): {val_metrics['class_0']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {val_metrics['class_0']['confusion_matrix'][1][0]}, True Positive (TP): {val_metrics['class_0']['confusion_matrix'][1][1]}")
    
    print(f"   Test set - Accuracy: {test_metrics['class_0']['accuracy']:.4f}, Precision: {test_metrics['class_0']['precision']:.4f}, Recall: {test_metrics['class_0']['recall']:.4f}, F1: {test_metrics['class_0']['f1']:.4f}, AUROC: {test_metrics['class_0']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {test_metrics['class_0']['confusion_matrix'][0][0]}, False Positive (FP): {test_metrics['class_0']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {test_metrics['class_0']['confusion_matrix'][1][0]}, True Positive (TP): {test_metrics['class_0']['confusion_matrix'][1][1]}")
    
    print("\nDeath prediction metrics:")
    print(f"   Training set - Accuracy: {train_metrics['class_1']['accuracy']:.4f}, Precision: {train_metrics['class_1']['precision']:.4f}, Recall: {train_metrics['class_1']['recall']:.4f}, F1: {train_metrics['class_1']['f1']:.4f}, AUROC: {train_metrics['class_1']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {train_metrics['class_1']['confusion_matrix'][0][0]}, False Positive (FP): {train_metrics['class_1']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {train_metrics['class_1']['confusion_matrix'][1][0]}, True Positive (TP): {train_metrics['class_1']['confusion_matrix'][1][1]}")
    
    print(f"   Validation set - Accuracy: {val_metrics['class_1']['accuracy']:.4f}, Precision: {val_metrics['class_1']['precision']:.4f}, Recall: {val_metrics['class_1']['recall']:.4f}, F1: {val_metrics['class_1']['f1']:.4f}, AUROC: {val_metrics['class_1']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {val_metrics['class_1']['confusion_matrix'][0][0]}, False Positive (FP): {val_metrics['class_1']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {val_metrics['class_1']['confusion_matrix'][1][0]}, True Positive (TP): {val_metrics['class_1']['confusion_matrix'][1][1]}")
    
    print(f"   Test set - Accuracy: {test_metrics['class_1']['accuracy']:.4f}, Precision: {test_metrics['class_1']['precision']:.4f}, Recall: {test_metrics['class_1']['recall']:.4f}, F1: {test_metrics['class_1']['f1']:.4f}, AUROC: {test_metrics['class_1']['auroc']:.4f}")
    print("   Confusion matrix:")
    print(f"     True Negative (TN): {test_metrics['class_1']['confusion_matrix'][0][0]}, False Positive (FP): {test_metrics['class_1']['confusion_matrix'][0][1]}")
    print(f"     False Negative (FN): {test_metrics['class_1']['confusion_matrix'][1][0]}, True Positive (TP): {test_metrics['class_1']['confusion_matrix'][1][1]}")
    
    print(f"\nExperiment completed: look_back={look_back}, shock_interval={shock_interval}, death_interval={death_interval}")
    print("="*80)
    
    return {
        'look_back': look_back,
        'shock_interval': shock_interval,
        'death_interval': death_interval,
        # Shock prediction metrics
        'train_shock_accuracy': train_metrics['class_0']['accuracy'],
        'train_shock_precision': train_metrics['class_0']['precision'],
        'train_shock_recall': train_metrics['class_0']['recall'],
        'train_shock_f1': train_metrics['class_0']['f1'],
        'train_shock_auroc': train_metrics['class_0']['auroc'],
        'val_shock_accuracy': val_metrics['class_0']['accuracy'],
        'val_shock_precision': val_metrics['class_0']['precision'],
        'val_shock_recall': val_metrics['class_0']['recall'],
        'val_shock_f1': val_metrics['class_0']['f1'],
        'val_shock_auroc': val_metrics['class_0']['auroc'],
        'test_shock_accuracy': test_metrics['class_0']['accuracy'],
        'test_shock_precision': test_metrics['class_0']['precision'],
        'test_shock_recall': test_metrics['class_0']['recall'],
        'test_shock_f1': test_metrics['class_0']['f1'],
        'test_shock_auroc': test_metrics['class_0']['auroc'],
        # Death prediction metrics
        'train_death_accuracy': train_metrics['class_1']['accuracy'],
        'train_death_precision': train_metrics['class_1']['precision'],
        'train_death_recall': train_metrics['class_1']['recall'],
        'train_death_f1': train_metrics['class_1']['f1'],
        'train_death_auroc': train_metrics['class_1']['auroc'],
        'val_death_accuracy': val_metrics['class_1']['accuracy'],
        'val_death_precision': val_metrics['class_1']['precision'],
        'val_death_recall': val_metrics['class_1']['recall'],
        'val_death_f1': val_metrics['class_1']['f1'],
        'val_death_auroc': val_metrics['class_1']['auroc'],
        'test_death_accuracy': test_metrics['class_1']['accuracy'],
        'test_death_precision': test_metrics['class_1']['precision'],
        'test_death_recall': test_metrics['class_1']['recall'],
        'test_death_f1': test_metrics['class_1']['f1'],
        'test_death_auroc': test_metrics['class_1']['auroc'],
        # Overall loss
        'train_loss': train_metrics['overall']['loss'],
        'val_loss': val_metrics['overall']['loss'],
        'test_loss': test_metrics['overall']['loss']
    }

def main():
    """Main function"""
    # Set random seed
    set_seed(42)
    
    # Load configuration
    config = Config()
    
    print("Loading data...")
    # Read large file in chunks
    chunks = []
    chunk_size = 100000  # Read 100,000 rows at a time
    
    print("Reading data_final.csv...")
    # Get file size to estimate total chunks
    file_size = os.path.getsize('./complete_data_and_labels/data_final.csv')
    estimated_chunks = file_size // (chunk_size * 100)  # Estimated number of chunks
    
    with tqdm(total=estimated_chunks, desc='Loading progress') as pbar:
        for chunk in pd.read_csv('./complete_data_and_labels/data_final.csv', chunksize=chunk_size):
            chunks.append(chunk)
            pbar.update(1)
    
    print("Merging data chunks...")
    raw_data = pd.concat(chunks, ignore_index=True)
    print(f"data_final.csv loaded, total {len(raw_data)} rows")
    
    print("\nReading shock_death_label...")
    raw_label = pd.read_csv(
        './complete_data_and_labels/shock_death_label_with_icu_deaths.csv',
        dtype={
            'stay_id': 'int32',
            'shock_time_interval': 'float32',
            'death_time_interval': 'float32'
        }
    )
    print(f"shock_death_label loaded, total {len(raw_label)} rows")
    
    # Create results directory
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    
    # Parameter combinations
    param_combinations = list(product(
        config.look_back_range,
        config.shock_interval_range,
        config.death_interval_range
    ))
    
    print("\n" + "="*80)
    print(f"Parameter search experiment started, total {len(param_combinations)} parameter combinations")
    print(f"Look-back window range: {list(config.look_back_range)}")
    print(f"Shock prediction window range: {list(config.shock_interval_range)}")
    print(f"Death prediction window range: {list(config.death_interval_range)}")
    print("="*80 + "\n")
    
    # Store results
    results = []
    
    # Run experiments
    experiment_pbar = tqdm(param_combinations, desc='Experiment progress')
    for idx, (look_back, shock_interval, death_interval) in enumerate(experiment_pbar):
        experiment_pbar.set_postfix({
            'look_back': look_back,
            'shock_interval': shock_interval,
            'death_interval': death_interval,
            'Experiment': f'{idx+1}/{len(param_combinations)}'
        })
        
        try:
            result = run_experiment(config, raw_data, raw_label, look_back, shock_interval, death_interval)
            results.append(result)
            
            # Save intermediate results
            save_results(results, config)
            
            print(f"\nCompleted: {idx+1}/{len(param_combinations)} parameter combinations")
            
        except Exception as e:
            print(f"\nExperiment failed: {str(e)}")
            continue
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to: {os.path.join(config.results_dir, f'{config.experiment_name}_results.csv')}")
    print("="*80)

if __name__ == "__main__":
    main() 