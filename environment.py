# This files serves as the environment for policy gradient estimators
# Games and gradient estimators are imlemented as instances of different classes

# TODO list:
# - performance plotting
# - resolve object issue

from reinforce import reinforce
from gpomdp import gpomdp
from svrpg import svrpg

from cart_pole import cart_pole

import numpy as np
import matplotlib.pyplot as plt

class Environment:
	def plot(self,game):
		titles = {
			cart_pole: 'Cart pole'
			#, mountain_car: 'Mountain car', lunar_rider: 'Lunar rider'
		}
		estimators = [
			reinforce.__name__
		]
		games = {
			cart_pole: 'cartpole'
		}
		data = [np.loadtxt('data--'+games[type(game)]+'_'+estim+'__CI-mean.txt') for estim in estimators]
		fig,ax = plt.subplots(figsize=(10,5))
		ax.set_title(titles[type(game)])
		ax.set_xlabel("episodes")
		ax.set_ylabel("reward")
		for i,item in enumerate(data):
			plt.plot(item[0])
			plt.fill_between(
				np.arange(item[0].shape[0]),
				item[0]-item[1],
			    item[0]+item[1],
			    alpha=0.2
			)
		fig.legend(
		    estimators
		)
		plt.grid()
		plt.savefig(titles[type(game)]+'.svg')
		plt.show()

	def train(self, estimator, game):
		# trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
		# of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)

		game.reset() # reset policy networks
		result = game.generate_data(estimator)

if __name__ == '__main__':
	environment = Environment()

	"""
	estimators = [
		#reinforce(),
		gpomdp()
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
	"""

	game = cart_pole()

	#estimator = reinforce(game)
	#estimator = gpomdp(game)
	estimator = svrpg(game)

	environment.train(estimator, game)
	environment.plot(game)
