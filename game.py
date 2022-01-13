"""
Interface for games  
"""

from abc import abstractmethod, ABCMeta


class game(metaclass=ABCMeta):
	def __init__(self):
		self.rewards_buffer = []
		self.number_of_sampled_trajectories = 0 # total number of sampled trajectories


	@abstractmethod
	def reset(self):
		pass


	abstractmethod
	def sample(self, max_t, eval):
		pass

	def generate_data(self, estimator, number_of_sampled_trajectories = 10000, number_of_runs = 30, root_path="./"):
		"""
		generate a file of 3d tuples: (number of sample trajectories, mean reward, CI)
		until it reaches the specified number of trajectories ("number_of_sampled_trajectories")
		"""
		results = []
		for _ in range(number_of_runs):
			self.reset()
			estimator_instance = estimator(self)
			# evaluations = []
			while True:
				estimator_instance.step(self) # performs one step of update for the selected estimator
									   # this can be one or more episodes
				# after policy NN updates, we need to evaluate this updated policy using self.evaluate()
				# evaluations.append((self.number_of_sampled_trajectories,self.evaluate()))
				# TODO: store the returned values: trajectories, mean_reward, CI_reward in some file
				if self.number_of_sampled_trajectories > number_of_sampled_trajectories:
					# print("finish",`${}`)
					print(f'finish run {_+1} of {number_of_runs}, length : {len(self.rewards_buffer)}, ntraj {self.number_of_sampled_trajectories}')
					self.number_of_sampled_trajectories = 0
					results.append(self.rewards_buffer)
					self.rewards_buffer = []
					break

		minLength = np.min([len(item) for item in results])
		for i in range(len(results)):
			results[i] = results[i][:minLength]
		# store a numpy binary
		name = 'data-v2--'+type(self).__name__+'_'+type(estimator_instance).__name__+'.npy'
		file_path = os.path.join(root_path, name)
		# store a numpy binary
		np.save(file_path,np.array(results))


	def evaluate(self): # performs the evaluation of the current policy NN
		# def evaluate(self, number_of_runs = 30):
		number_of_sampled_trajectories = self.number_of_sampled_trajectories
		results = self.sample(200,eval=1)['rewards']
		# results = [np.sum(self.sample(200, eval = 1)['rewards']) for i in range(number_of_runs)]
		self.number_of_sampled_trajectories = number_of_sampled_trajectories

		# TODO:
		# it should return 3 values:
		# 1) self.number_of_sampled_trajectories
		# 2) mean performance
		# 3) confidence interval
		return np.sum(results)
		# return (self.number_of_sampled_trajectories,np.mean(results),np.std(results))
