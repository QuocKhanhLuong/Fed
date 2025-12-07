"""
Checkpoint Manager for Federated Learning

Handles saving and loading of model checkpoints:
- Best model (highest accuracy)
- Periodic checkpoints (every N rounds)
- Latest checkpoint (for resume)

Author: Research Team
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints for Federated Learning.
    
    Saves:
    - best_model.npz: Best model by accuracy
    - latest.npz: Most recent model
    - round_{N}.npz: Periodic checkpoints
    - metadata.json: Training metadata
    
    Usage:
        manager = CheckpointManager("./checkpoints")
        
        # Save during training
        manager.save_checkpoint(weights, round_num, accuracy)
        
        # Load for inference or resume
        weights = manager.load_best_model()
    """
    
    def __init__(
        self, 
        checkpoint_dir: str = "./checkpoints",
        save_frequency: int = 5,  # Save every N rounds
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save periodic checkpoint every N rounds
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.best_accuracy = 0.0
        self.best_round = 0
        self.checkpoints_saved = 0
        
        # Metadata tracking
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'best_accuracy': 0.0,
            'best_round': 0,
            'total_rounds': 0,
            'checkpoints': [],
        }
        
        # Load existing metadata if available
        self._load_metadata()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def _load_metadata(self):
        """Load metadata from disk if exists."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.best_accuracy = self.metadata.get('best_accuracy', 0.0)
                    self.best_round = self.metadata.get('best_round', 0)
                logger.info(f"Loaded existing metadata: best_acc={self.best_accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def save_checkpoint(
        self,
        weights: List[np.ndarray],
        round_num: int,
        accuracy: float,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: Optional[bool] = None,
    ) -> Dict[str, str]:
        """
        Save model checkpoint.
        
        Args:
            weights: Model weights as numpy arrays
            round_num: Current round number
            accuracy: Global accuracy
            metrics: Additional metrics to save
            is_best: Force best model save (auto-detect if None)
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        # Prepare checkpoint data
        checkpoint = {
            'weights': weights,
            'round': round_num,
            'accuracy': accuracy,
            'metrics': metrics or {},
            'saved_at': datetime.now().isoformat(),
        }
        
        # 1. Save latest (always)
        latest_path = self.checkpoint_dir / "latest.npz"
        self._save_weights(checkpoint, latest_path)
        saved_files['latest'] = str(latest_path)
        
        # 2. Check if best
        if is_best is None:
            is_best = accuracy > self.best_accuracy
        
        if is_best:
            self.best_accuracy = accuracy
            self.best_round = round_num
            best_path = self.checkpoint_dir / "best_model.npz"
            self._save_weights(checkpoint, best_path)
            saved_files['best'] = str(best_path)
            logger.info(f"🏆 New best model saved: acc={accuracy:.4f} (round {round_num})")
        
        # 3. Periodic checkpoint
        if round_num > 0 and round_num % self.save_frequency == 0:
            periodic_path = self.checkpoint_dir / f"round_{round_num}.npz"
            self._save_weights(checkpoint, periodic_path)
            saved_files['periodic'] = str(periodic_path)
            logger.info(f"📦 Periodic checkpoint saved: round_{round_num}.npz")
        
        # Update metadata
        self.metadata['best_accuracy'] = self.best_accuracy
        self.metadata['best_round'] = self.best_round
        self.metadata['total_rounds'] = round_num
        self.metadata['checkpoints'].append({
            'round': round_num,
            'accuracy': accuracy,
            'is_best': is_best,
        })
        self._save_metadata()
        
        self.checkpoints_saved += 1
        return saved_files
    
    def _save_weights(self, checkpoint: Dict, path: Path):
        """Save checkpoint to npz file."""
        try:
            # Convert weights list to dict for npz
            weights_dict = {f'layer_{i}': w for i, w in enumerate(checkpoint['weights'])}
            
            # Save weights
            np.savez_compressed(
                path,
                **weights_dict,
                _metadata=np.array([json.dumps({
                    'round': checkpoint['round'],
                    'accuracy': checkpoint['accuracy'],
                    'metrics': checkpoint['metrics'],
                    'saved_at': checkpoint['saved_at'],
                    'num_layers': len(checkpoint['weights']),
                })])
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary with weights and metadata
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        try:
            data = np.load(path, allow_pickle=True)
            
            # Extract metadata
            metadata = json.loads(str(data['_metadata'][0]))
            num_layers = metadata['num_layers']
            
            # Extract weights in order
            weights = [data[f'layer_{i}'] for i in range(num_layers)]
            
            logger.info(f"Loaded checkpoint: {path.name}, acc={metadata['accuracy']:.4f}")
            
            return {
                'weights': weights,
                'round': metadata['round'],
                'accuracy': metadata['accuracy'],
                'metrics': metadata['metrics'],
            }
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def load_best_model(self) -> List[np.ndarray]:
        """Load best model weights."""
        best_path = self.checkpoint_dir / "best_model.npz"
        if not best_path.exists():
            raise FileNotFoundError("No best model checkpoint found")
        
        checkpoint = self.load_checkpoint(str(best_path))
        return checkpoint['weights']
    
    def load_latest(self) -> Dict[str, Any]:
        """Load latest checkpoint for resume."""
        latest_path = self.checkpoint_dir / "latest.npz"
        if not latest_path.exists():
            raise FileNotFoundError("No latest checkpoint found")
        
        return self.load_checkpoint(str(latest_path))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get checkpoint summary."""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_accuracy': self.best_accuracy,
            'best_round': self.best_round,
            'checkpoints_saved': self.checkpoints_saved,
            'save_frequency': self.save_frequency,
        }


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("=" * 60)
    print("Checkpoint Manager - Unit Test")
    print("=" * 60)
    
    # Create temp directory
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize manager
        manager = CheckpointManager(test_dir, save_frequency=2)
        
        # Simulate 5 rounds
        for round_num in range(1, 6):
            # Dummy weights
            weights = [np.random.randn(100, 100).astype(np.float32)]
            accuracy = 0.5 + round_num * 0.05 + np.random.rand() * 0.02
            
            # Save checkpoint
            saved = manager.save_checkpoint(weights, round_num, accuracy)
            print(f"Round {round_num}: acc={accuracy:.4f}, saved={list(saved.keys())}")
        
        print(f"\n✓ Best accuracy: {manager.best_accuracy:.4f} (round {manager.best_round})")
        
        # Load best model
        best_weights = manager.load_best_model()
        print(f"✓ Loaded best model: {len(best_weights)} layers")
        
        # Load latest
        latest = manager.load_latest()
        print(f"✓ Loaded latest: round {latest['round']}, acc={latest['accuracy']:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    finally:
        shutil.rmtree(test_dir)
