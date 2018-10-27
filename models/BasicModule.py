import torch as t
import time

class BasicModule(t.nn.Module):
	"""docstring for BasicModule"""
	def __init__(self):
		super(BasicModule, self).__init__()
		self.model_name = str(type(self))

	def load(self, path):
		self.load_state_dict(t.load(path))

	def save(self, name=None):
		if name is None:
			prefix = 'checkpoints/' + self.model_name + '_'
			name = time.strftime(prefix + '.pth')
		t.save(self.state_dict(), name)
		return name

	def get_optimizer(self, lr, weight_decay):
		return t.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)

class Flat(t.nn.Module):
	"""docstring for Flat"""
	def __init__(self):
		super(Flat, self).__init__()
	
	def forward(self, x):
		return x.view(x.size(0), -1)
		