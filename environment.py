"""
This files serves as the environment for policy gradient estimators
Games and gradient estimators are imlemented as instances of different classes
"""

from reinforce import Reinforce
from gpomdp import Gpomdp
from svrpg import Svrpg
from svrpg_manual_adam import Svrpg_manual
from svrpg_automatic_adam import SvrpgAuto
from sarahpg import SarahPg
from stormpg import StormPg
from pagepg import PagePg
from pagestormpg import PageStormPg

from cart_pole import cart_pole
from continuous_mountain_car import continuous_mountain_car
from mountain_car import mountain_car
from lunar_lander import lunar_lander

import numpy as np
import matplotlib.pyplot as plt

import argparse
class Environment:
    def registerEstimatorClasses(self, items):
        """
			items : either a list of items or a single `item`
			item  : {'slug':slug,'class':Estimator_Child}
			slug  : string, slug of the class, has to be unique
			class : a class herited from "Estimator"
		"""
        # test wether it is a single item or a list of items
        if type(items).__name__ != 'list':
            items = [items]
        # if estimators property exists
        if not hasattr(self, 'estimators'):
            self.estimators = {}
        for estimator in items:
            if hasattr(self.estimators, estimator['slug']):
                raise Exception('estimation slug already used, has to be unique')
            self.estimators[estimator['slug']] = estimator['class']

    def registerGameInstances(self, items):
        """
			items : either a list of items or a single `item`
			item  : {'slug':slug,'plotTitle':title,'instance':game_child}
			slug  : string, slug of the game, has to be unique
			plotTitle: string, a title for the plot
			instance : an instance of a game
		"""
        # test wether it is a single item or a list of items
        if type(items).__name__ != 'list':
            items = [items]

        # if estimators property exists
        if not hasattr(self, 'games'):
            self.games = {}
        for game in items:
            if hasattr(self.games, game['slug']):
                raise Exception('game slug already used, has to be unique')
            self.games[game['slug']] = {
                'plotTitle': game['plotTitle'],
                'instance': game['instance'],
            }

    def plot_old(self, game, estimators='all'):
        game = self.games[game]
        data = []
        maxReward = 200
        # estimator can be either 'all', a string (single estim)
        # or a list
        if estimators == 'all':
            estimators = self.estimators
        elif type(estimators).__name__ == 'str':
            estimators = {
                estimators: self.estimators[estimators]
            }
        else:
            estimators = {
                slug: self.estimators[slug]
                for slug in estimators
            }

        # load all demanded estimators
        for key, estimator in estimators.items():
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
                np.maximum(item[1] - item[2],0),
                np.minimum(item[1] + item[2],maxReward),
                alpha=0.2
            )
        fig.legend(frameon=False, loc='upper center', ncol=len(data))
        plt.grid()
        plt.savefig(game['plotTitle'] + '.svg')
        plt.show()
    def plot(self, game, estimators='all',interval=1):
        game = self.games[game]
        data = []
        maxReward = 200
        # estimator can be either 'all', a string (single estim)
        # or a list
        if estimators == 'all':
            estimators = self.estimators
        elif type(estimators).__name__ == 'str':
            estimators = {
                estimators: self.estimators[estimators]
            }
        else:
            estimators = {
                slug: self.estimators[slug]
                for slug in estimators
            }
        # load all demanded estimators
        for key, estimator in estimators.items():
            try:
                file = np.load(
                    'data-runs--'
                    + type(game['instance']).__name__
                    + '_' + estimator.__name__
                    + '.npy',allow_pickle=True
                )
                data.append({
                    'content': file[:,:,1],
                    'slug': key,
					'indexes':file[:,:,0]
                })
            except FileNotFoundError:
                print(estimator.__name__)
                print(f'no generated data for {estimator}')
                continue
        # top 10 rewards
        for i,value in enumerate(data):
            indexes = (-np.sum(value['content'],axis=1)).argsort()
            data[i]['content']=value['content'][indexes[:10],::interval]
            data[i]['indexes']=value['indexes'][indexes[:10],::interval]
        # compute statistics
        for i,value in enumerate(data):
            mean = value['content'].mean(axis=0)
            std = value['content'].std(axis=0)
            data[i]['content'] = [data[i]['indexes'][0,:],mean,std]
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
                np.maximum(item[1] - item[2],0),
                np.minimum(item[1] + item[2],maxReward),
                alpha=0.2
            )
        fig.legend(frameon=False, loc='upper center', ncol=len(data))
        plt.grid()
        plt.savefig(game['plotTitle'] + '.svg')
        plt.show()

    def train(self, estimator, game, num_traj=1000, reps=20, output_path=""):
        # trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
        # of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)
        game = self.games[game]['instance']
        # estimator = self.estimators[estimator](game)
        game.reset()  # reset policy networks
        print(f"Starting training of {game} with {reps}x {num_traj} trajectories")
        result = game.generate_data(self.estimators[estimator],num_traj,reps, output_path)


if __name__ == '__main__':


    environment = Environment()

    environment.registerEstimatorClasses([
        {'slug': 'Reinforce', 'class': Reinforce},
        {'slug': 'Gpomdp', 'class': Gpomdp},
        {'slug': 'SarahPg', 'class': SarahPg},
        {'slug': 'PageStormPg', 'class': PageStormPg},
        {'slug': 'Svrpg', 'class': SvrpgAuto},
        {'slug': 'StormPg', 'class': StormPg},
        {'slug': 'PagePg', 'class': PagePg},
        {'slug': 'Svrpg_auto', 'class': SvrpgAuto},
    ])
    environment.registerGameInstances([
        {'slug': 'cart_pole', 'plotTitle': 'Cart pole', 'instance': cart_pole()},
        {'slug': 'lunar_lander', 'plotTitle': 'Lunar Lander', 'instance': lunar_lander()},
        {'slug': 'continuous_mountain_car', 'plotTitle': 'Continuous mountian car', 'instance': continuous_mountain_car()},
        {'slug': 'mountain_car', 'plotTitle': ' mountian car', 'instance': mountain_car()}
    ])


    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, choices=list(environment.games.keys()), help="Game to be tested", default="cart_pole")
    parser.add_argument("--estimator", type=str, choices=list(environment.estimators.keys()) + ["all"], help="Estimator to be used", default="Gpomdp")
    parser.add_argument("--output", type=str, help="Output directory path", default="./")
    parser.add_argument("--num_traj", type=int,  help="Number of Total Trajectories", default=10000)
    parser.add_argument("--iter", type=int,  help="Number of repeted iterations", default=20)
    parser.add_argument("--plot", action="store_true",
                    help="Plot the given estimator")
    parser.add_argument("--use_cuda", action="store_true",
                    help="Use CUDA")
    args = parser.parse_args()
    

    if args.plot:
        environment.plot(estimators=args.estimator, game=args.game)
    else:
        environment.train(estimator=args.estimator, game=args.game, reps=args.iter, num_traj=args.num_traj, output_path=args.output)
