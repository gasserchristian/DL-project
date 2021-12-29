# This files serves as the environment for policy gradient estimators
# Games and gradient estimators are imlemented as instances of different classes

# TODO list:
# - performance plotting

from reinforce import reinforce
from cart_pole import cart_pole

class Environment:
	def plot(self,game):
		titles = {
			cart_pole: 'Cart pole'
			#, mountain_car: 'Mountain car', lunar_rider: 'Lunar rider'
		}
		estimators = [
			type(reinforce).__name__
		]
		games = {
			cart_pole: 'cartpole'
		}
		data = [np.loadtxt('data--'+games[game]+'_'+estim.__name__+'.txt') for estim in estimators]
		fig,ax = plt.subplots(figsize=(10,5))
		ax.set_title(titles[game])
		ax.set_xlabel("episodes")
		ax.set_ylabel("reward")
		for i,item in enumerate(data):
			plt.plot(data[i][1,:])
			plt.fill_between(
				np.arange(data[i][1,:].shape[0]),
				data[i][1,:]-data[i][2,:],
			    data[i][1,:]+data[i][2,:],
			    alpha=0.2
			)
		fig.legend(
		    estimators
		)
		plt.grid()
		plt.savefig(titles[game]+'.svg')
		plt.show()

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
