from abc import abstractmethod, ABCMeta

class estimator(metaclass=ABCMeta):
	@abstractmethod
	def step(self):
		"""
		computes policy loss
		"""
		pass
