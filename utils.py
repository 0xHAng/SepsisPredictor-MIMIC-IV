import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_time_string(time_str):
    """Process abnormal time string"""
    try:
        # If NaN or empty value, return NaT
        if pd.isna(time_str) or time_str == '':
            return pd.NaT
            
        # Check if it's in "YYYY-MM-DD" format
        if len(str(time_str).split()) == 1:
            # Add time part
            time_str = f"{time_str} 00:00:00"
        else:
            # Return original value (may already have time part), let to_datetime handle it
            return time_str
            
        return time_str
    except:
        # If there are other exceptions, return NaT
        return pd.NaT

def process_data(raw_data, raw_label, config, look_back, shock_interval, death_interval, delete_columns=None):
    
    min_records = int((look_back + 1) * 0.5)
    ABPm_null_ratio = config.ABPm_null_ratio
    
    """Process raw data"""
    if delete_columns is None:
        delete_columns = ['stay_id', 'charttime', 'antibiotic_time', 'sepsis_time', 'look_back_begin', 'look_back_end']
        
    print("1. Copying data...")
    data = raw_data.copy()
    label = raw_label.copy()
    
    print("2. Converting time format...")
    # Time format conversion
    data['charttime'] = pd.to_datetime(data['charttime'])
    data['antibiotic_time'] = data['antibiotic_time'].apply(process_time_string)
    data['antibiotic_time'] = pd.to_datetime(data['antibiotic_time'])
    data['sepsis_time'] = data['sepsis_time'].apply(process_time_string)
    data['sepsis_time'] = pd.to_datetime(data['sepsis_time'])
    
    print("3. Filtering based on data...")
    with tqdm(total=6, desc="Data filtering progress") as pbar:
        label = raw_label[(raw_label['shock_time_interval'] >= look_back) | (raw_label['shock_time_interval'].isnull())].copy()
        label = label[(label['death_time_interval'] >= look_back) | (label['death_time_interval'].isnull())].copy()

        label['sepsis_time'] = pd.to_datetime(label['sepsis_time'])

        # Calculate whether shock or death occurs within shock_interval and death_interval
        label['has_shock_interval'] = label['shock_time_interval'] <= shock_interval + look_back
        label['has_death_interval'] = label['death_time_interval'] <= death_interval + look_back

        label['label'] = label[['has_shock_interval', 'has_death_interval']].apply(lambda x: [int(x.iloc[0]), int(x.iloc[1])], axis=1)

        # Set stay_id as index
        label.set_index('stay_id', inplace=True)
        pbar.update(1)

        # (1) Require data to start no later than 1 hour after sepsis_time
        # Count records for each stay_id
        data_stay_count = data.groupby('stay_id').size().reset_index(name='record_count').set_index("stay_id")

        data_time = data.groupby('stay_id').agg({
            'charttime':['min', 'max'],
        })

        data_time.columns = ['charttime_min', 'charttime_max']
        data_time = pd.merge(data_time, label["sepsis_time"], left_index=True, right_index=True)
        data_time = pd.merge(data_time, data_stay_count, left_index=True, right_index=True)

        data_time['chart_after_in'] = (data_time['charttime_min'] - data_time['sepsis_time']).dt.total_seconds() / 3600
        print("Currently total samples: ", data_time.shape[0])

        data_time = data_time[data_time['chart_after_in'] < 1]
        print("【Condition】: charttime cannot lag sepsis_time too much, remaining total samples:", data_time.shape[0])
        pbar.update(1)

        # (2) Require data to end time to be greater than or equal to the end time of lookback
        # Round to the nearest hour
        data_time['look_back_begin'] = data_time['sepsis_time']
        data_time['look_back_end'] = (data_time['sepsis_time'] + pd.Timedelta(hours=look_back))
        # data_time['interval_end'] = (data_time['sepsis_time'] + pd.Timedelta(hours=look_back + interval)).dt.floor('h')
        data_time = data_time[data_time['charttime_max'] >= data_time['look_back_end']]
        print("【Condition】: Maximum charttime later than (greater than) look_back_end, remaining total samples:", data_time.shape[0])
        pbar.update(1)

        # (3) Extract data from lookback time period from data
        filtered_data = pd.merge(data, data_time[['look_back_begin', 'look_back_end']], left_on='stay_id', right_index=True, how='inner')
        filtered_data = filtered_data[(filtered_data['charttime'] >= filtered_data['look_back_begin']) & (filtered_data['charttime'] <= filtered_data['look_back_end'])]
        print("【Processed】: charttime in lookback, remaining total samples:", filtered_data['stay_id'].nunique())
        pbar.update(1)

        # (4) Require data to have at least min_records records in the lookback interval
        record_count = filtered_data.groupby('stay_id').size().reset_index(name='record_count')
        record_count = record_count[record_count['record_count'] >= min_records]
        data_time = data_time[data_time.index.isin(record_count['stay_id'])]
        filtered_data = filtered_data[filtered_data['stay_id'].isin(record_count['stay_id'])]
        print("【Condition】: lookback records count not less than min_records, remaining total samples:", filtered_data['stay_id'].nunique())
        pbar.update(1)

        # (5) Process, so that the time span of each sample is consistent
        # Determine time range between look_back_begin and look_back_end
        all_time_ranges = []
        for index, row in tqdm(data_time.iterrows()):
            all_time_ranges.append(pd.date_range(row['look_back_begin'], row['look_back_end'], freq='1h'))
        # Organize all time ranges into DataFrame
        expanded_time_df = pd.DataFrame({
            'stay_id': np.repeat(data_time.index, [len(tr) for tr in all_time_ranges]),
            'charttime': np.concatenate(all_time_ranges)
        })
        test_expanded_time_df = expanded_time_df.groupby('stay_id').size().reset_index(name='record_count')
        assert (test_expanded_time_df['record_count'] == look_back + 1).all()

        filtered_data = pd.merge(expanded_time_df, filtered_data, on=['stay_id', 'charttime'], how='left').sort_values(by=['stay_id', 'charttime'])
        # filtered_data['stay_id_tmp'] = filtered_data['stay_id']
        # filtered_data = filtered_data.groupby('stay_id_tmp').ffill().bfill()

        assert (filtered_data.groupby('stay_id').size() == look_back + 1).all()
        print("【Ensure】In filtered_data, each stay_id has {} records.".format(look_back + 1))

        filtered_data.drop(columns=['look_back_begin', 'look_back_end'], inplace=True)

        # (6) Require filtered_data to have empty value ratio
        # Count ABPm and uri empty value ratio for each stay_id

        ABPm_null_count = filtered_data.groupby('stay_id')['mbp'].apply(lambda x: x.isnull().sum()).reset_index(name='ABPm_null_count')
        ABPm_null_count['ABPm_null_ratio'] = ABPm_null_count['ABPm_null_count'] / (look_back + 1)
        ABPm_null_count = ABPm_null_count[ABPm_null_count['ABPm_null_ratio'] <= ABPm_null_ratio]
        print("【Condition】: ABPm empty value ratio not greater than ABPm_null_ratio, remaining total samples:", ABPm_null_count.shape[0])

        after_rui_abpm_stay_ids = set(ABPm_null_count['stay_id'])
        filtered_data_result = filtered_data[filtered_data['stay_id'].isin(after_rui_abpm_stay_ids)]
        print("【Condition】: Empty value ratio all meet requirements, remaining total samples:", filtered_data_result['stay_id'].nunique())

        # # Fill each stay_id in filtered_data_result with
        filtered_data_result['stay_id_tmp'] = filtered_data_result['stay_id'].copy()
        # filtered_data_result = filtered_data_result.groupby('stay_id_tmp').ffill().bfill()
        # Fill 0
        filtered_data_result.fillna(0, inplace=True)

        # Finally, filter data from label based on filtered_data_result's stay_id
        label = label[label.index.isin(filtered_data_result['stay_id'])]
        pbar.update(1)
    print(f"Data filtering completed! Remaining total samples: {filtered_data_result['stay_id'].nunique()}")
    
    return filtered_data_result, label

def split_data(label, test_size=0.2, random_state=42):
    """Split dataset"""
    print("\nStart splitting dataset...")
    with tqdm(total=2, desc="Dataset splitting") as pbar:
        # First split: training set and (validation+test) set
        train_label, val_test_label = train_test_split(
            label, 
            test_size=test_size,
            random_state=random_state,
            stratify=label['label']
        )
        pbar.update(1)
        
        # Second split: validation set and test set
        test_label, val_label = train_test_split(
            val_test_label,
            test_size=1/3,
            random_state=random_state,
            stratify=val_test_label['label']
        )
        pbar.update(1)
    
    print(f"Dataset splitting completed!")
    print(f"Training set: {len(train_label)} samples")
    print(f"Validation set: {len(val_label)} samples")
    print(f"Test set: {len(test_label)} samples")
    
    return train_label, val_label, test_label

class ShockDeathDataset(Dataset):
    """Dataset class"""
    def __init__(self, label, data, delete_columns):
        print("\nCreating dataset...")
        self.data = data.fillna(0)
        self.label = label
        self.stay_ids = list(label.index)
        self.feature_columns = [col for col in data.columns if col not in delete_columns]
        
        print("Standardizing features...")
        # Standardize features
        with tqdm(total=len(self.feature_columns), desc="Feature standardization") as pbar:
            for col in self.feature_columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / (
                    self.data[col].std() if self.data[col].std() != 0 
                    else 1 if self.data[col].mean() != 0 
                    else 1
                )
                pbar.update(1)
        
        print(f"Dataset created! Feature count: {len(self.feature_columns)}")
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        X = self.data[self.data['stay_id'] == stay_id]
        X = X[self.feature_columns].values
        Y = self.label.loc[stay_id, 'label']
        
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0.0
    predictions, true_labels = [], []
    outputs_proba = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluation progress", leave=False):
            inputs, labels = [d.to(device) for d in data]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions.extend((outputs > 0.5).int().cpu().numpy())
            outputs_proba.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(data_loader)
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    outputs_proba = np.array(outputs_proba)
    
    # Calculate metrics for each class
    metrics = {}
    
    for i in range(true_labels.shape[1]):
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels[:, i], predictions[:, i])
        
        metrics[f'class_{i}'] = {
            'accuracy': accuracy_score(true_labels[:, i], predictions[:, i]),
            'precision': precision_score(true_labels[:, i], predictions[:, i], zero_division=0),
            'recall': recall_score(true_labels[:, i], predictions[:, i]),
            'f1': f1_score(true_labels[:, i], predictions[:, i]),
            'auroc': roc_auc_score(true_labels[:, i], outputs_proba[:, i]),
            'confusion_matrix': cm
        }
    
    # Calculate overall metrics
    metrics['overall'] = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(true_labels, predictions, average='macro'),
        'f1': f1_score(true_labels, predictions, average='macro'),
        'auroc': roc_auc_score(true_labels, outputs_proba, average='macro'),
        'loss': val_loss
    }
    
    return metrics

def save_results(results, config):
    """Save experiment results"""
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
        
    results_df = pd.DataFrame(results)
    results_path = os.path.join(config.results_dir, f'{config.experiment_name}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}") 