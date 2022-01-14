# DL-project

## Structure 
- environment.py - main script that generates data for different games/estimators and plot it  
- game.py - interface for games
- estimator.py - interface for estimators
- reinforce.py - reinforce estimator 
- cart_pole.py - cart-pole game 


## Members

- Vasilii Kopylov vkopylov@student.ethz.ch
- Jakub Mandula jmandula@student.ethz.ch
- Emiljo Mehillaj emehillaj@student.ethz.ch
- Christian Gasser chgasser@student.ethz.ch

Brainstorming doc : https://docs.google.com/document/d/1ORe8bYeyK1qdgkFnvwzRyqKiPX85h4kklDUvl_49D6g/edit#

## Proposal

Overleaf link: [https://www.overleaf.com/project/61b07a2d5c52e4b1697888ec](https://www.overleaf.com/project/61b07a2d5c52e4b1697888ec)


# Running the code

```bash
# Install the requirements
pip install -r requirements.txt


# Run the given estimator on a game with custom parameters for 1000 trajectories and 20 repetitions
python main.py --game lunar_lander --estimator PagePg --prob 0.3 --iter 20 --num_traj 1000 --output ./runs

# Plot a given run file
python main.py --plot_files ./runs/run1.npy



# See this for more details
python main.py -h
usage: main.py [-h] [--game {cart_pole,lunar_lander,continuous_mountain_car,mountain_car,pendulum}] [--estimator {Reinforce,Gpomdp,SarahPg,PageStormPg,Svrpg,StormPg,PagePg,all}]
                      [--output OUTPUT] [--num_traj NUM_TRAJ] [--iter ITER] [--subit SUBIT] [--batch_size BATCH_SIZE] [--mini_batch_size MINI_BATCH_SIZE] [--flr FLR] [--lr LR] [--mlr MLR]
                      [--prob PROB] [--alpha ALPHA] [--plot_files PLOT_FILES [PLOT_FILES ...]] [--plot] [--use_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --game {cart_pole,lunar_lander,continuous_mountain_car,mountain_car,pendulum}
                        Game to be tested
  --estimator {Reinforce,Gpomdp,SarahPg,PageStormPg,Svrpg,StormPg,PagePg,all}
                        Estimator to be used
  --output OUTPUT       Output directory path
  --num_traj NUM_TRAJ   Number of Total Trajectories
  --iter ITER           Number of repeted iterations
  --subit SUBIT         Max allowed number of subiterations
  --batch_size BATCH_SIZE
                        Batch Size
  --mini_batch_size MINI_BATCH_SIZE
                        Mini Batch Size
  --flr FLR             First Learning rate
  --lr LR               Learning rate
  --mlr MLR             this is magnitude of update by self.optimizer_sub
  --prob PROB           Probability
  --alpha ALPHA         Alpha
  --plot_files PLOT_FILES [PLOT_FILES ...]
                        Plot Specific Files
  --plot                Plot the given estimator
  --use_cuda            Use CUDA




```


## Schedule

| Date       | Comment                                              |
| ---------- | ---------------------------------------------------- |
| 11.12.2021 | Submit 1-page proposal, bonus 0.25 on proposal grade |
| 20.12.2021 | Submit 1-page proposal                               |
| 04.01.2021 | 5-page report+code, bonus 0.25                       |
| 14.01.2021 | 5-page report+code without bonus                     |

## Grading

- project is 30% of DL class
  - 10% of the project grade is the proposal



## Further informations

- project information on intro slides, page 9
- grading criteria on [Deep Learning 2021 (ethz.ch)](http://www.da.inf.ethz.ch/teaching/2021/DeepLearning/) at the end of the page
- how to, for write report on [howto-paper.pdf (ethz.ch)](http://www.da.inf.ethz.ch/teaching/2021/DeepLearning/files/howto-paper.pdf)

## Git

**Commands for pushing**

```shell
git clone git@github.com:gasserchristian/DL-project.git
cd DL-project
git commit -m "message"
git pull
git push

# list remote branches
git branch -r
# list local branches, 
git branch

# first fetch if want to checkout to a remote branch
git fetch
git checkout <branch name>

# Merge branchA into branchB and continue on branchB
git fetch
git checkout branchB
git merge branchA
```

Cheatsheet on: [git-cheat-sheet.pdf (gitlab.com)](https://about.gitlab.com/images/press/git-cheat-sheet.pdf)

