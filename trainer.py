"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze import Maze
from sarsa_brain import SarsaTable as S_Brain
from q_learning_brain import QLearningTable as Q_Brain
from deep_q_brain import DeepQNetwork as DQ_Brain
import json



class Trainer():

    def __init__(
        self,
        config_file,
        value_sets_file,
        alg="Q"
        ):
        with open(value_sets_file) as json_data_file:
            self.value_sets = json.load(json_data_file)

        with open(config_file) as json_data_file:
            self.config = json.load(json_data_file)

        self.train(alg)

    def train(self, alg):

        maze = Maze(self.config, self.value_sets, 'save.csv', 'cache.json', 20, 60);
        if(alg=='Q'):
            RL = Q_Brain(actions=list(range(maze.n_actions)), learning_rate=0.05)
        if(alg=='DQ'):
            RL = DQ_Brain(actions=list(range(maze.n_actions)), learning_rate=0.05, replace_target_iter=50)
        elif(alg=='S'):
            RL = S_Brain(actions=list(range(maze.n_actions)), learning_rate=0.05)


        for episode in range(500):
            # initial observation
            observation = maze.reset(rand=False)
            action = RL.choose_action(str(observation))
            first = True

            while True:

                if first:
                    print("####### new round started #######")
                    print("start observation: " + str(observation))

                # RL take action and get next observation and reward
                observation_, reward, done = maze.step(action)

                # RL choose action based on next observation
                action_ = RL.choose_action(str(observation_))

                if(alg=='Q'):
                    # RL learn from this transition (s, a, r, s) ==> Q-lening
                    RL.learn(str(observation), action, reward, str(observation_))
                elif(alg=='DQ'):
                    RL.store_transition(str(observation), action, reward, str(observation_))
                    RL.learn()
                elif(alg=='S'):
                    # RL learn from this transition (s, a, r, s, a) ==> Sarsa
                    RL.learn(str(observation), action, reward, str(observation_), action_)

                # swap observation and action
                observation = observation_
                action = action_

                # break while loop when end of this episode
                if done:
                    break

        # end of game
        print('game over')
