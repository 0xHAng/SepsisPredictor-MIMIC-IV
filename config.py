import argparse

class Config:
    def __init__(self):
        # Data parameters
        self.look_back_range = range(11, 36, 6)  # Look-back window range [12,24]
        self.shock_interval_range = range(12, 25, 6)  # Shock prediction window range [12,24]
        self.death_interval_range = range(168, 750, 24)  # Death prediction window range [168,744]
        self.min_records_ratio = 0.5  # Minimum records ratio
        self.ABPm_null_ratio = 0.2  # ABPm null value ratio threshold
        
        # Model parameters
        self.d_model = 64
        self.nhead = 8
        self.num_encoder_layers = 2
        self.dim_feedforward = 256
        self.dropout = 0.1
        
        # Training parameters
        self.batch_size = 32
        self.lr = 0.001
        self.num_epochs = 100  # Set a larger value, rely on early stopping to control training epochs
        self.lr_factor = 0.5
        self.lr_patience = 3
        self.early_stopping = 6  # Increase early stopping epochs to avoid premature stopping
        
        # Experiment recording parameters
        self.device = 'cuda'
        self.experiment_name = 'parameter_search_test'  # Modify experiment name
        self.checkpoint_dir = './checkpoints'
        self.results_dir = './results'
        
    def get_model_config(self, look_back, W):
        return {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'input_size': W,
            'look_back': look_back
        } 