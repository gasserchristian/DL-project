"""
This files serves as the environment for policy gradient estimators
Games and gradient estimators are imlemented as instances of different classes
"""

import numpy as np
import matplotlib.pyplot as plt

import argparse
import re
import os

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
                item[1] - item[2],
                np.minimum(item[1] + item[2],maxReward),
                alpha=0.2
            )
        fig.legend(frameon=False, loc='upper center', ncol=len(data))
        plt.grid()
        plt.savefig(game['plotTitle'] + '.svg')
        plt.show()

    def plot_by_file(self, files, interval=1):
        games = {}
        for f in files:
            name = f.name
            if not name.endswith('.npy'):
                print(f"{name}Not a result file")

            game = None
            for g in list(self.games.keys()):
                if g in name:
                    game = g
                    break
            if game == None:
                print(f"Couldnt get game of {name}")
                continue

            estimator = None
            for e in list(self.estimators.keys()):
                if e in name:
                    estimator = e
                    break

            if estimator == None:
                print(f"Couldnt get estimator of {name}")
                continue

            # Extract parameters:

            reg = re.search(r"__([0-9]+)__([0-9]+)_([0-9]+)_(((\w+):(.+)_)|((\w+):(.+)-))?", name)
            if reg == None:
                print(f"Failed to parse prameters of {name}")
                trajectores = None
                iters = None
                batch = None
                sweep_name = None
                sweep_value = None
            else:

                captured = reg.groups()

                trajectories = int(captured[0])
                iters = int(captured[1])
                batch = int(captured[2])

                sweep_name = captured[5]
                sweep_value = None
                if sweep_name == None:
                    sweep_name = captured[8]
                    if sweep_name != None:
                        sweep_value = captured[9]
                else:
                    sweep_value = captured[6]
                print(f"Parsed {name} with game {game} est {estimator} {iters}x{trajectories} batch:{batch} {sweep_name}: {sweep_value}")
            
            if game not in games:
                games[game] = {}
            if estimator not in games[game]:
                games[game][estimator] = []

            raw_data = np.load(name)[:,:]

            indexes = (-np.sum(raw_data,axis=1)).argsort()
            best=raw_data[indexes[:10],::interval]

            mean = best.mean(axis=0)
            std = best.std(axis=0)
            statistics = np.array([np.arange(len(mean))*10,mean,std])

            # if np.any(statistics[1,-40:] < 100) or np.any(statistics[1,-10:] < 125):
            #     print(f"Best score not met {statistics[1,-1:]}")
            #     continue

            # if estimator == "PagePg":
            #     if np.any(statistics[1,-45:] < 100) or not "alpha:0.7" in name or np.any(statistics[1,-20:] < 125) :
            #         continue

            

            if sweep_name != None:
                extra_name = f"_{sweep_name}:{sweep_value}"
            else:
                extra_name = ""
            data = {
                "raw_data": raw_data,
                'statistics': statistics,
                'extra_name': extra_name,
                'file_name': name[10:]

            }

            games[game][estimator].append(data)


        maxReward = 200

        # Plot 1 grph per game
        for game_name, estimators in games.items():

            title = f"Trajectory score of {game_name} environment"
            if game_name == "lunar_lander":
                factor = 0.1
            else:
                factor= 1
            

            fig, ax = plt.subplots(figsize=(8, 6))
            # ax.set_title(title)
            ax.set_xlabel("Number of Trajectories")
            ax.set_ylabel("Total Reward")
            for estimator_name, estimator in estimators.items():

                
                if estimator_name in ["PagePg", "PageStormPg"]:
                    alpha=1.0
                    linewidth=2.5

                else:
                    alpha=0.5
                    linewidth=1
                ax = plt.gca()
                for run in estimator:
                    item = run['statistics']
                    item[0] *= factor
                    
                    cut = -1
                    x = item[0][:cut]
                    y = item[1][:cut]
                    std =item[2][:cut]
                    label = estimator_name #+ run['file_name']
                    color=next(ax._get_lines.prop_cycler)['color']
                    plt.plot(x, y, label=label, alpha=alpha, linewidth=linewidth, color=color)
                    if game_name != "lunar_lander":
                        plt.scatter(x, y, s=100, alpha=alpha*0.8, edgecolor=color, facecolor="none")
                    plt.fill_between(
                        x,
                        y - std,
                        np.minimum(y + std,maxReward),
                        alpha=0.2*alpha
                    )
            ax.legend(frameon=True, loc='best')
            plt.tight_layout()
            plt.grid()
            plt.savefig(title + '.svg')
            plt.savefig(title + '.png')
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
                    'data-v2--'
                    + type(game['instance']).__name__
                    + '_' + estimator.__name__
                    + '.npy'
                )
                data.append({
                    'content': file[:,:],
                    'slug': key
                })
            except FileNotFoundError:
                print(estimator.__name__)
                print(f'no generated data for {estimator}')
                continue
        # top 10 rewards
        for i,value in enumerate(data):
            indexes = (-np.sum(value['content'],axis=1)).argsort()
            data[i]['content']=value['content'][indexes[:10],::interval]
        # compute statistics
        for i,value in enumerate(data):
            mean = value['content'].mean(axis=0)
            std = value['content'].std(axis=0)
            data[i]['content'] = [np.arange(len(mean))*10,mean,std]
        # plot the curves
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(game['plotTitle'])
        ax.set_xlabel("trajectories")
        ax.set_ylabel("reward")



        for i, value in enumerate(data):
            item = value['content']
            label = value['slug']
            print(label)
            if label in ["PagePg", "PageStormPg"]:
                alpha=1
                print(label, 'string')
            else:
                alpha=0.6
            plt.plot(item[0], item[1], label=label, alpha=alpha)
            plt.fill_between(
                item[0],
                np.maximum(item[1] - item[2],0),
                np.minimum(item[1] + item[2],maxReward),
                alpha=0.2*alpha
            )
        fig.legend(frameon=False, loc='upper center', ncol=len(data))
        plt.grid()
        plt.savefig(game['plotTitle'] + '.svg')
        plt.show()

    def train(self, estimator, game, args, sweep_parameter,hyper_parameters, number_of_sampled_trajectories = 10000, number_of_runs = 30, output_path=""):
        # num_traj=1000, reps=20, output_path="")
        # trains the chosen estimator on the selected RL game and generates the results as a CSV file consisting
        # of following 3d tuples: (number of trajectories, average return, 90% confidence bounds)
        print(f"Starting training of {game} using {estimator} with {number_of_runs}x {number_of_sampled_trajectories} trajectories")
        game = self.games[game]['instance']
        # estimator = self.estimators[estimator](game)
        game.reset()  # reset policy networks
        result = game.generate_data(self.estimators[estimator],hyper_parameters,sweep_parameter, number_of_sampled_trajectories,number_of_runs,output_path)

