from mimo.model.preprocess import main as preprocess
from mimo.model.train import main as train
import os

if not os.path.exists('dataset.pt'):
	preprocess()

train()
