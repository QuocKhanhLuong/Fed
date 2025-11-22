#!/bin/bash
# Test script for real dataset integration
# Run this after installing dependencies: pip install -r requirements.txt

set -e

echo "=================================="
echo "FL-QUIC-LoRA Dataset Integration Test"
echo "=================================="

# Test 1: Data Manager with CIFAR-100
echo -e "\n[TEST 1] Testing data_manager.py with CIFAR-100..."
python -c "
from client.data_manager import load_dataset
import logging
logging.basicConfig(level=logging.INFO)

print('Loading CIFAR-100 with Dirichlet partitioning...')
train_loader, val_loader, test_loader, stats = load_dataset(
    dataset_name='cifar100',
    data_dir='./data',
    client_id=0,
    num_clients=5,
    partition_type='dirichlet',
    alpha=0.5,
    batch_size=32,
)
print(f'✓ Dataset stats: {stats}')

# Check batch
for images, labels in train_loader:
    print(f'✓ Batch shape: {images.shape}')
    print(f'✓ Label range: [{labels.min()}, {labels.max()}]')
    break
"

# Test 2: Model Trainer with real data
echo -e "\n[TEST 2] Testing model_trainer.py with advanced metrics..."
python -c "
from client.model_trainer import MobileViTLoRATrainer
from client.data_manager import load_dataset
import logging
logging.basicConfig(level=logging.INFO)

print('Creating trainer with LoRA...')
trainer = MobileViTLoRATrainer(
    num_classes=100,
    lora_r=4,
    use_sam=False,
    use_tta=False,
)

print('Loading CIFAR-100 dataset...')
train_loader, val_loader, test_loader, stats = load_dataset(
    dataset_name='cifar100',
    data_dir='./data',
    client_id=0,
    num_clients=5,
    batch_size=32,
)

print('Training for 1 epoch...')
metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
print(f'✓ Training metrics: {metrics}')

print('Evaluating with advanced metrics...')
eval_metrics = trainer.evaluate(test_loader)
print(f'✓ Evaluation metrics: {eval_metrics}')
print(f'  - Accuracy: {eval_metrics[\"accuracy\"]:.4f}')
if 'f1_macro' in eval_metrics:
    print(f'  - F1-Score (Macro): {eval_metrics[\"f1_macro\"]:.4f}')
if 'auroc' in eval_metrics:
    print(f'  - AUROC: {eval_metrics[\"auroc\"]:.4f}')
"

# Test 3: FL Client with real data
echo -e "\n[TEST 3] Testing fl_client.py integration..."
python -c "
from client.fl_client import create_fl_client
from client.data_manager import load_dataset
import logging
logging.basicConfig(level=logging.INFO)

print('Loading dataset...')
train_loader, val_loader, test_loader, stats = load_dataset(
    dataset_name='cifar100',
    data_dir='./data',
    client_id=0,
    num_clients=5,
    batch_size=32,
)

print('Creating FL client...')
client = create_fl_client(
    num_classes=stats['num_classes'],
    lora_r=4,
    local_epochs=1,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

print('Getting initial parameters...')
params = client.get_parameters({})
print(f'✓ Extracted {len(params)} parameter arrays')

print('Simulating FL training round...')
config = {'round': 1, 'local_epochs': 1, 'learning_rate': 1e-3}
updated_params, num_samples, metrics = client.fit(params, config)
print(f'✓ Training complete: {num_samples} samples')
print(f'  - Loss: {metrics[\"loss\"]:.4f}')
print(f'  - Accuracy: {metrics[\"accuracy\"]:.4f}')
"

echo -e "\n=================================="
echo "✅ ALL TESTS PASSED!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run FL client: python client/app_client.py --dataset cifar100 --alpha 0.5"
echo "2. Enable SAM: python client/app_client.py --use-sam --dataset cifar100"
echo "3. Enable TTA: python client/app_client.py --use-tta --dataset cifar100"
echo "4. Try MedMNIST: python client/app_client.py --dataset pathmnist --alpha 0.3"
echo ""
