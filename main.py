
#!/usr/bin/env python


import sys
sys.path.append('src')


from environment import Environment


from reinforce import Reinforce
from gpomdp import Gpomdp
from svrpg import Svrpg
from sarahpg import SarahPg
from stormpg import StormPg
from pagepg import PagePg
from pagestormpg import PageStormPg

from cart_pole import cart_pole
from continuous_mountain_car import continuous_mountain_car
from mountain_car import mountain_car
from lunar_lander import lunar_lander
from pendulum import pendulum



import argparse
import os


if __name__ == '__main__':


    environment = Environment()

    environment.registerEstimatorClasses([
        {'slug': 'Reinforce', 'class': Reinforce},
        {'slug': 'Gpomdp', 'class': Gpomdp},
        {'slug': 'SarahPg', 'class': SarahPg},
        {'slug': 'PageStormPg', 'class': PageStormPg},
        {'slug': 'Svrpg', 'class': Svrpg},
        {'slug': 'StormPg', 'class': StormPg},
        {'slug': 'PagePg', 'class': PagePg},
    ])

    environment.registerGameInstances([
        {'slug': 'cart_pole', 'plotTitle': 'Cart pole', 'instance': cart_pole()},
        {'slug': 'lunar_lander', 'plotTitle': 'Lunar Lander', 'instance': lunar_lander()},
        {'slug': 'continuous_mountain_car', 'plotTitle': 'Continuous mountian car', 'instance': continuous_mountain_car()},
        {'slug': 'mountain_car', 'plotTitle': ' Mountian car', 'instance': mountain_car()},
        {'slug': 'pendulum', 'plotTitle': 'Pendulum', 'instance': pendulum()}
    ])


    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, choices=list(environment.games.keys()), help="Game to be tested", default="cart_pole")
    parser.add_argument("--estimator", type=str, choices=list(environment.estimators.keys()) + ["all"], help="Estimator to be used", default="Gpomdp")
    parser.add_argument("--output", type=str, help="Output directory path", default="./")
    parser.add_argument("--num_traj", type=int,  help="Number of Total Trajectories", default=500)
    parser.add_argument("--iter", type=int,  help="Number of repeted iterations", default=10)

    parser.add_argument("--subit", type=int,  help="Max allowed number of subiterations")
    parser.add_argument("--batch_size", type=int,  help="Batch Size")
    parser.add_argument("--mini_batch_size", type=int,  help="Mini Batch Size")
    parser.add_argument("--flr", type=float,  help="First Learning rate")
    parser.add_argument("--lr", type=float,  help="Learning rate")
    parser.add_argument("--mlr", type=float,  help="this is magnitude of update by self.optimizer_sub")
    parser.add_argument("--prob", type=float,  help="Probability")
    parser.add_argument("--alpha", type=float,  help="Alpha")

    parser.add_argument("--plot_files", type=argparse.FileType('r'), nargs='+', help="Plot Specific Files")
    parser.add_argument("--plot", action="store_true",
                    help="Plot the given estimator")
    parser.add_argument("--use_cuda", action="store_true",
                    help="Use CUDA")

    args = parser.parse_args()




    default_hyper_parameters = {
            "subit": 3,
            "batch_size": 5,
            "mini_batch_size": 3,
            "flr": 6e-3,
            "lr": 3e-3,
            "mlr": 1,
            "prob":0.9,
            "alpha":0.9
        }

    estimator_hyper_parameters  = {
        "Reinforce": {"mini_batch_size":10, "lr": 3e-3},
        "Gpomdp": {"mini_batch_size":5, "lr": 3e-3},
        "SarahPg": {"batch_size": 5, "mini_batch_size": 3, "lr": 3e-2, "flr": 6e-2},
        "PageStormPg": {"batch_size": 5, "mini_batch_size": 3, "lr": 3e-3},
        "StormPg": {"batch_size": 5, "mini_batch_size": 3, "lr": 3e-3},
        "PagePg": {"lr": 2.5e-2, "batch_size": 5, "mini_batch_size": 3, "flr": 6e-3},
        "Svrpg": {"batch_size": 5, "mini_batch_size": 3, "lr": 3e-3, "flr": 6e-3},
        "all": {}
    }

    configured_hyper_parameters = {
        "subit": args.subit,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "flr": args.flr,
        "lr": args.lr,
        "mlr": args.mlr,
        "prob": args.prob,
        "alpha": args.alpha
    }
    configured_hyper_parameters = {k: v for k, v in configured_hyper_parameters.items() if v is not None}

    sweep_parameter= "_".join([f"{v}:{configured_hyper_parameters[v]}" for v in configured_hyper_parameters])

    hyper_parameters = {**default_hyper_parameters, **estimator_hyper_parameters[args.estimator], **configured_hyper_parameters}


    if args.plot_files:
        environment.plot_by_file(args.plot_files)
    elif args.plot:
        environment.plot(estimators=args.estimator, game=args.game)
    else:
        print("Hyper parameters:")
        print(hyper_parameters)
        environment.train(estimator=args.estimator, game=args.game, args=args, sweep_parameter=sweep_parameter ,hyper_parameters=hyper_parameters, number_of_runs=args.iter, number_of_sampled_trajectories=args.num_traj, output_path=args.output)
