# This files serves as the environment for policy gradient estimators
# Games and gradient estimators are imlemented as instances of different classes

# TODO list:
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
	def registerEstimatorClasses(self,items):
		"""
			items : either a list of items or a single `item`
			item  : {'slug':slug,'class':Estimator_Child}
			slug  : string, slug of the class, has to be unique
			class : a class herited from "Estimator"
		"""
		# test wether it is a single item or a list of items
		if type(items).__name__!='list':
			items = [items]
		# if estimators property exists
		if not hasattr(self,'estimators'):
			self.estimators={}
		for estimator in items:
			if hasattr(self.estimators,estimator['slug']):
				raise Exception('estimation slug already used, has to be unique')
			self.estimators[estimator['slug']] = estimator['class']
	def registerGameInstances(self,items):
		"""
			items : either a list of items or a single `item`
			item  : {'slug':slug,'plotTitle':title,'instance':game_child}
			slug  : string, slug of the game, has to be unique
			plotTitle: string, a title for the plot
			instance : an instance of a game
		"""
		# test wether it is a single item or a list of items
		if type(items).__name__!='list':
			items = [items]

		# if estimators property exists
		if not hasattr(self,'games'):
			self.games={}
		for game in items:
			if hasattr(self.games,game['slug']):
				raise Exception('game slug already used, has to be unique')
			self.games[game['slug']] = {
				'plotTitle':game['plotTitle'],
				'instance':game['instance'],
			}
	def plot(self, game, estimators='all'):
		game = self.games[game]
		data = []

		# estimator can be either 'all', a string (single estim)
		# or a list
		if estimators == 'all':
			estimators = self.estimators
		elif type(estimators).__name__=='str':
			estimators = {
				estimators: self.estimators[estimators]
			}
		else:
			estimators = {
				slug:self.estimators[slug]
				for slug in estimators
			}

		# load all demanded estimators
		for key,estimator in estimators.items():
			try:
				data.append({
					'content': np.loadtxt(
						'data--'
						+ type(game['instance']).__name__
						+ '_' + estimator.__name__
						+ '.txt'
					),
					'slug': key
				})
			except FileNotFoundError:
				print(estimator.__name__)
				print(f'no generated data for {estimator}')
				continue

		# plot the curves
		fig, ax = plt.subplots(figsize=(10, 5))
		ax.set_title(game['plotTitle'])
		ax.set_xlabel("trajectories")
		ax.set_ylabel("reward")
		for i, value in enumerate(data):
			item = value['content']
			label = value['slug']
			plt.plot(item[0], item[1], label=label)
			plt.fill_between(
                item[0],
                item[1] - item[2],
                item[1] + item[2],
                alpha=0.2
            )
		fig.legend(frameon=False, loc='upper center', ncol=len(data))
		plt.grid()
		plt.savefig(game['plotTitle'] + '.svg')
		plt.show()

	def train(self, estimator, game):
		# trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
		# of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)
		game = self.games[game]['instance']
		estimator = self.estimators[estimator](game)
		game.reset()  # reset policy networks
		print("Starting training")
		result = game.generate_data(estimator)


if __name__ == '__main__':
    environment = Environment()

    environment.registerEstimatorClasses([
    	{'slug':'Reinforce','class':Reinforce},
    	{'slug':'Gpomdp','class':Gpomdp},
    	{'slug':'SarahPg','class':SarahPg},
    	{'slug':'PageStormPg','class':PageStormPg},
    	{'slug':'Svrpg','class':Svrpg},
    	{'slug':'StormPg','class':StormPg},
    	{'slug':'PagePg','class':PagePg},
    ])
    environment.registerGameInstances([
    	{'slug':'cart_pole','plotTitle':'Cart pole','instance':cart_pole()}
    ])

    # environment.train(estimator='Reinforce',game='cart_pole')
    # environment.train(estimator='Gpomdp',game='cart_pole')
    # environment.train(estimator='SarahPg',game='cart_pole')
    # environment.train(estimator='PageStormPg',game='cart_pole')
    # environment.train(estimator='Svrpg',game='cart_pole')
    # environment.train(estimator='StormPg',game='cart_pole')
    # environment.train(estimator='PagePg',game='cart_pole')

    # environment.plot('cart_pole',estimators=['StormPg','SarahPg'])
    environment.plot('cart_pole',estimators='all')
    # environment.plot('cart_pole',estimators='SarahPg')
