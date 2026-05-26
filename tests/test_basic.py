import os
import pytest
import torch
import numpy as np
from src.model import EdgeRouterNet, ArcMarginProduct
from src.dataset import AllCropsDS
from src.config import NUM_CLASSES

def test_model_init():
    """Test that the model can be initialized and handle a forward pass."""
    model = EdgeRouterNet(kernels=(3, 5, 7))
    xb = torch.randn(2, 64, 256) # (batch, channels, length)
    len_l, sh_l, lo_l, emb = model(xb)
    
    assert len_l.shape == (2, 2)
    assert sh_l.shape == (2, 2)
    assert lo_l.shape == (2, 3)
    assert emb.shape[1] == model.emb_dim

def test_arcface_init():
    """Test that ArcMarginProduct can be initialized."""
    arc = ArcMarginProduct(in_features=128, out_features=5)
    emb = torch.randn(2, 128)
    label = torch.tensor([0, 1])
    output = arc(emb, label)
    assert output.shape == (2, 5)

def test_dataset_mock_gen():
    """Test that mock data can be generated and files exist."""
    from src.generate_mock_data import generate_mock_data
    test_dir = "./data/test_mock"
    os.makedirs(test_dir, exist_ok=True)
    generate_mock_data(test_dir, n_subjects=1)
    
    files = os.listdir(test_dir)
    assert any(f.endswith(".mat") for f in files)
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
