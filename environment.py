# This files serves as the environment for policy gradient estimators
# Games and gradient estimators are imlemented as instances of different classes

# TODO list:
# - performance plotting

from reinforce import reinforce
from cart_pole import cart_pole

class Environment:
	def plot(self,game):
		# TODO: plot the perfomance for all estimators on the selected game using the data from CSV files
		pass
		'''
		mean = np.loadtxt('pathMean.txt')
		std = np.loadtxt('pathStd.txt')
		fig,ax = plt.subplots(figsize=(10,10))
		ax.set_title("placeholder")
		ax.set_xlabel("epoch")
		ax.set_ylabel("Reward")
		for i in range(5):
			plt.plot(mean[i])
			plt.fill_between(
				mean[i]-std[i],
			    mean[i]+std[i],
			    alpha=0.2
			)
		fig.legend(
		    [
		        'placeholder',
		        'placeholder',
		        'placeholder',
		        'placeholder',
		        'placeholder'
		    ]
		)
		plt.grid()
		plt.savefig('dummy-name.svg')

		# QUESTIONS :
		- title
		- save or show ?
		- game argument -> text ?
		'''

	def train(self, estimator, game):
		# trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
		# of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)
		result = game.generate_data(estimator)

if __name__ == '__main__':
	environment = Environment()

	estimators = [
		reinforce()
		#gpomdp(),
		#storm(),
		#page()
		]

	games = [
		cart_pole()
		#mountain_car(),
		#lunar_rider()
		]

	print("Starting training")
	for estimator in estimators:
		for game in games:
			environment.train(estimator, game)

	print("Plotting the performance")
	for game in games:
		environment.plot(game)
