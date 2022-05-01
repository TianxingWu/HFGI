from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import dlib
import numpy as np
from torch.utils.data import dataloader

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


class FakeNewsDataset(Dataset):

	def __init__(self, image_paths, predictor_path, opts, transform=None, preprocess=None):
		self.paths = image_paths
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
			if from_im is None:
				# print("no faces detected!")
				return None
		else:
			raise ValueError("no preprocess method!")
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, np.array(orig_img), quad, crop, pad


def safe_collate_fn(batch):
	# when batch_size larger than 1, skip no face items. But have bug when all items in one batch are None.
	# Can not use because of json metadata writing issues.
	# batch = list(filter(lambda x: x is not None, batch))
	# return dataloader.default_collate(batch)

	# Only support batch_size=1
	if batch[0] is None:
		return None
	else:
		return dataloader.default_collate(batch)