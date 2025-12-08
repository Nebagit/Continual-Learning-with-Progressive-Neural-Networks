# Configuration file for my Progressive Neural Network (PNN) experiments

# Number of flows in a batch for each column
flow_size = 100

# Number of packets per flow
pkt_size = 200

# Number of flows each column will learn from at a time
column_batch_size = 128

# Batch size used during training within each column
learning_batch_size = 16

# Base batch sizes for experimentation / reference
base_batch_sizes = [4, 8, 16, 32, 64, 128]

# Learning rate for optimizer
learning_rate = 1e-3

# Number of epochs for column training
epochs = 50

# Number of epochs for task-specific training (if needed)
task_epochs = 20

# Labels used for testing the model
test_labels = ['benign', 'attack_bot', 'attack_DDOS', 'attack_portscan']

# All labels available for training / experimentation
all_labels = [
    'vectorize_friday/benign',
    'attack_bot',
    'attack_DDOS',
    'attack_portscan',
    'Benign_Wednesday',
    'DOS_SlowHttpTest',
    'DOS_SlowLoris',
    'DOS_Hulk',
    'DOS_GoldenEye',
    'FTPPatator',
    'SSHPatator',
    'Web_BruteForce',
    'Web_XSS'
]
