"""
Interface for games  
"""

from abc import abstractmethod, ABCMeta


class game(metaclass=ABCMeta):
	@abstractmethod
	def generate_data(self, estimator, number_of_runs):
		"""
		generate a collection of 3d tuples (episode, average reward, variance)
		for the given estimator for the "number_of_runs" runs 
		"""
		pass


