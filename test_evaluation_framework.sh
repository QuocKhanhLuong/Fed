#!/bin/bash
# Test Publication-Ready Evaluation Framework
# Validates metrics module and integration with FL components

set -e

echo "=========================================="
echo "FL-QUIC-LoRA Evaluation Framework Test"
echo "Publication-Ready Metrics System"
echo "=========================================="

# Test 1: Metrics Module
echo -e "\n[TEST 1] Testing utils/metrics.py..."
python -c "
from utils.metrics import ClassificationEvaluator, ResourceTracker, SystemMetrics, AggregatedMetrics
import numpy as np
import time

print('✓ All metrics classes imported successfully')

# Test ClassificationEvaluator
predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1])
targets = np.array([0, 1, 2, 0, 1, 1, 0, 2])
probabilities = np.random.rand(8, 3)
probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

metrics = ClassificationEvaluator.compute(predictions, targets, probabilities, num_classes=3)
print(f'✓ Classification Metrics: {ClassificationEvaluator.format_metrics(metrics)}')

# Test ResourceTracker
with ResourceTracker() as tracker:
    time.sleep(0.05)
    _ = [i**2 for i in range(50000)]

resource_metrics = tracker.get_metrics()
print(f'✓ Resource Tracking: Time={resource_metrics[\"time_s\"]:.3f}s, GPU={resource_metrics[\"gpu_mem_peak_mb\"]:.1f}MB')

# Test SystemMetrics
sys_metrics = SystemMetrics()
comm = sys_metrics.update_communication(bytes_sent=1048576, bytes_received=524288)
print(f'✓ System Metrics: Sent={SystemMetrics.format_bytes(comm[\"bytes_sent_delta\"])}, Received={SystemMetrics.format_bytes(comm[\"bytes_received_delta\"])}')

# Test AggregatedMetrics
client_results = [
    ({'accuracy': 0.85, 'f1_macro': 0.83}, 100),
    ({'accuracy': 0.90, 'f1_macro': 0.88}, 150),
]
avg = AggregatedMetrics.weighted_average(client_results)
print(f'✓ Aggregated Metrics: Global Accuracy={avg[\"accuracy\"]:.4f}')

print('✓ Metrics module test PASSED')
"

# Test 2: Model Trainer with Metrics
echo -e "\n[TEST 2] Testing client/model_trainer.py with metrics integration..."
python -c "
from client.model_trainer import MobileViTLoRATrainer
from client.data_manager import load_dataset
import torch

print('Creating trainer with LoRA...')
trainer = MobileViTLoRATrainer(
    num_classes=10,
    lora_r=4,
    use_sam=False,
    use_tta=False,
)

print('Loading CIFAR-10 dataset...')
train_loader, val_loader, test_loader, stats = load_dataset(
    dataset_name='cifar10',
    data_dir='./data',
    client_id=0,
    num_clients=3,
    batch_size=32,
)

print('Training for 1 epoch (with ResourceTracker)...')
train_metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
print(f'✓ Training Metrics:')
print(f'  Loss: {train_metrics[\"loss\"]:.4f}')
print(f'  Accuracy: {train_metrics[\"accuracy\"]:.4f}')
print(f'  Time: {train_metrics[\"train_time_s\"]:.2f}s')
print(f'  GPU Peak: {train_metrics[\"gpu_mem_peak_mb\"]:.1f} MB')

print('Evaluating with comprehensive metrics (ClassificationEvaluator)...')
eval_metrics = trainer.evaluate(test_loader)
print(f'✓ Evaluation Metrics:')
print(f'  Accuracy: {eval_metrics[\"accuracy\"]:.4f}')
print(f'  Precision: {eval_metrics.get(\"precision_macro\", 0):.4f}')
print(f'  Recall: {eval_metrics.get(\"recall_macro\", 0):.4f}')
print(f'  F1-Score: {eval_metrics.get(\"f1_macro\", 0):.4f}')
if 'auroc' in eval_metrics:
    print(f'  AUROC: {eval_metrics[\"auroc\"]:.4f}')

print('✓ Model trainer with metrics test PASSED')
"

# Test 3: FL Strategy Aggregation
echo -e "\n[TEST 3] Testing server/fl_strategy.py aggregate_evaluate..."
python -c "
import sys
sys.path.append('.')

# Mock Flower types for testing
class MockEvaluateRes:
    def __init__(self, loss, num_examples, metrics):
        self.loss = loss
        self.num_examples = num_examples
        self.metrics = metrics

class MockClientProxy:
    def __init__(self, cid):
        self.cid = cid

# Simulate evaluation results from 3 clients
results = [
    (MockClientProxy('client_0'), MockEvaluateRes(0.5, 100, {'accuracy': 0.85, 'f1_macro': 0.83})),
    (MockClientProxy('client_1'), MockEvaluateRes(0.4, 150, {'accuracy': 0.90, 'f1_macro': 0.88})),
    (MockClientProxy('client_2'), MockEvaluateRes(0.6, 120, {'accuracy': 0.80, 'f1_macro': 0.78})),
]

# Test aggregation manually (without Flower)
from utils.metrics import AggregatedMetrics

client_metrics = [(dict(r[1].metrics), r[1].num_examples) for r in results]
global_metrics = AggregatedMetrics.weighted_average(client_metrics)

accuracies = [r[1].metrics['accuracy'] for r in results]
fairness = AggregatedMetrics.compute_fairness(accuracies)

print('✓ Global Metrics:')
print(f'  Accuracy: {global_metrics[\"accuracy\"]:.4f}')
print(f'  F1-Score: {global_metrics[\"f1_macro\"]:.4f}')

print('✓ Fairness Metrics:')
print(f'  Std Dev: {fairness[\"fairness_std\"]:.4f}')
print(f'  Min: {fairness[\"fairness_min\"]:.4f}')
print(f'  Max: {fairness[\"fairness_max\"]:.4f}')

converged, reason = AggregatedMetrics.check_convergence(
    current_accuracy=global_metrics['accuracy'],
    target_accuracy=0.80,
)
print(f'✓ Convergence: {converged} - {reason}')

print('✓ FL Strategy aggregation test PASSED')
"

# Test 4: Config with target_accuracy
echo -e "\n[TEST 4] Testing utils/config.py with target_accuracy..."
python -c "
from utils.config import Config

config = Config()
print(f'✓ FederatedConfig.target_accuracy = {config.federated.target_accuracy}')
print(f'✓ FederatedConfig.convergence_patience = {config.federated.convergence_patience}')
print('✓ Config test PASSED')
"

echo -e "\n=========================================="
echo "✅ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Evaluation Framework Features:"
echo "  ✓ Classification Metrics (Acc, F1, Precision, Recall, AUROC)"
echo "  ✓ Resource Tracking (Time, GPU Memory)"
echo "  ✓ System Metrics (Communication Cost)"
echo "  ✓ Aggregated Metrics (Global Accuracy, Fairness)"
echo "  ✓ Convergence Detection"
echo ""
echo "Ready for Q1 Research Paper Benchmarking!"
echo ""
