# This files serves as the environment for policy gradient estimators
# Games and gradient estimators are imlemented as instances of different classes

# TODO list:
# - performance plotting
# - resolve object issue

from reinforce import Reinforce
from gpomdp import Gpomdp
from svrpg import Svrpg
from sarahpg import SarahPg
from stormpg import StormPg
from pagepg import PagePg
from pagestormpg import PageStormPg
from cart_pole import cart_pole

import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def plot(self, game):
        titles = {
            cart_pole: 'Cart pole'
            # , mountain_car: 'Mountain car', lunar_rider: 'Lunar rider'
        }
        estimators = [
            Reinforce.__name__,
            Gpomdp.__name__,
            Svrpg.__name__
        ]
        games = {
            cart_pole: 'cartpole'
        }
        data = []
        for estimator in estimators:
            try:
                data.append(np.loadtxt('data--' + games[type(game)] + '_' + estimator + '.txt'))
            except:
                print(f'no generated data for {estimator}')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(titles[type(game)])
        ax.set_xlabel("trajectories")
        ax.set_ylabel("reward")
        for i, item in enumerate(data):
            plt.plot(item[0], item[1], label=estimators[i])
            plt.fill_between(
                item[0],
                item[1] - item[2],
                item[1] + item[2],
                alpha=0.2
            )

        fig.legend(frameon=False, loc='upper center', ncol=len(data))
        # ax.legend(
        #     estimators,artists
        # )
        plt.grid()
        plt.savefig(titles[type(game)] + '.svg')
        plt.show()

    def train(self, estimator, game):
        # trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
        # of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)

        game.reset()  # reset policy networks
        result = game.generate_data(estimator)


if __name__ == '__main__':
    environment = Environment()

    """
	estimators = []
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

    # estimator = Reinforce(game)
    # estimator = Gpomdp(game)
    estimator = PagePg(game)
    environment.train(estimator, game)
    environment.plot(game)
