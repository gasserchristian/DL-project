from abc import abstractmethod, ABCMeta

class estimator(metaclass=ABCMeta):
	@abstractmethod
	def compute_loss(self, trajectory):
		"""
		computes policy loss
		"""
		pass
