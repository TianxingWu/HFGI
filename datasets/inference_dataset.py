from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import dlib
import numpy as np


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, preprocess=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im

class InTheWildDataset(Dataset):

	def __init__(self, root, predictor_path, opts, transform=None, preprocess=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts
		self.predictor = dlib.shape_predictor(predictor_path)
		self.detector = dlib.get_frontal_face_detector()
    	

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			orig_img = Image.open(from_path).convert('RGB')
			from_im, quad, crop, pad = self.preprocess(orig_img, self.predictor, self.detector)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, np.array(orig_img), quad, crop, pad
