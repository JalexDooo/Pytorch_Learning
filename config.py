import torch as t
import warnings

class DefaultConfig(object):
	"""docstring for DefaultConfig"""
	env = 'default' # visdom
	vis_port = 8097
	model = 'AlexNet'

	train_data_root = './data/train/'
	test_data_root = './data/test'
	load_model_path = None

	batch_size = 32
	use_gpu = True
	num_workers = 4 # how many workers for loading data
	print_freq = 20 # print every N batch

	debug_file = './tmp/debug'
	result_file = './result/result.csv'

	max_epoch = 10
	lr = 0.1
	lr_decay = 0.95
	weight_decay = 1e-4

	device = t.device('cuda') if use_gpu else t.device('cpu')

	def _parse(self, kwargs):
		"""
		update config
		"""
		for k, v in kwargs.items():
			if not hasattr(self, k):
				warnings.warn("Warning: opt has not attribute %s" %k)
			setattr(self, k, v)

		# opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

		print('user config:')

		for k, v in self.__class__.__dict__.items():
			if not k.startswith('_'):
				print(k, getattr(self, k))

opt = DefaultConfig()

"""
opt = DefaultConfig()
new_config = {
	'batch_size':20,
	'use_gpu':False,
}
opt._parse(new_config)
print(opt.batch_size)
print(opt.use_gpu)
"""