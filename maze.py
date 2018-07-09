
import numpy as np
import time
import sys

from neural_nets import mnist

import random

import simplejson as json
import codecs
import ast

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start



class Maze():
    def __init__(
        self,
        config,
        value_sets,
        save_file,
        cache_file,
        steps_per_round,
        epochs
        ):


        self.config = config;
        self.initial_config = config.copy()
        self.value_sets = value_sets;
        self.save_file = save_file;
        self.cache_file = cache_file;
        self.steps_per_round = steps_per_round;

        self.neuro = mnist.Mnist(config, value_sets)

        with open(cache_file) as json_data_file:
            self.cache = json.load(json_data_file)

        self.dimensions = list(self.value_sets.keys())
        #garant same sort of dimensions each time
        self.dimensions = sorted(self.dimensions, reverse=True)
        self.action_space = []
        self.position = {}
        for index in range(len(self.dimensions)):
            self.action_space.append([self.dimensions[index], 'up'])
            self.action_space.append([self.dimensions[index], 'down'])
            self.position[self.dimensions[index]] = self.config[self.dimensions[index]]



        self.n_actions = len(self.action_space)
        self.n_features = len(self.dimensions)

        self.neuro.set_config(self.config)

        self.epochs = epochs

        self.current_accuracy, self.current_loss = self.cache_train()

        self.step_count = 0

    def reset(self, rand=True):

        #config net with random settings
        for index in range(len(self.dimensions)):
            dim = self.dimensions[index]
            min_val = self.value_sets[dim]["min"]
            max_val = self.value_sets[dim]["max"]
            step_size = self.value_sets[dim]["step"]

            if rand:
                new_val = randrange_float(min_val, max_val, step_size)
                if(type(self.config[dim]) is int): # if value started as an int set it as an int
                    new_val = int(new_val)
                self.config[dim] = new_val
            else:
                self.config = self.initial_config.copy()

            #reset the position object to config
            self.position[dim] = self.config[dim]

        self.neuro.set_config(self.config)

        return np.array(list(self.position.values()))

    def get_from_cache(self, position):
        for key in self.cache.keys():
            if position == ast.literal_eval(key):
                if "visited" in self.cache[key]:
                    self.cache[key]["visited"] += 1
                else:
                    self.cache[key]["visited"] = 1
                json.dump(self.cache, codecs.open(self.cache_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
                return self.cache[key]


        return None

    def cache_train(self):
        cache_vals = self.get_from_cache(self.position)
        if(cache_vals != None):
            #cache_vals = self.cache[str(self.position)];
            accuracy = np.float32(cache_vals["acc"])
            loss = np.float32(cache_vals["loss"])
        else:
            accuracy0, loss0 = self.neuro.train(self.epochs)
            accuracy1, loss1 = self.neuro.train(self.epochs)
            accuracy2, loss2 = self.neuro.train(self.epochs)
            accuracy3, loss3 = self.neuro.train(self.epochs)
            accuracy = (accuracy0 + accuracy1 + accuracy2 + accuracy3) / 4
            loss = (loss0 + loss1 + loss2 + loss3) / 4
            self.cache[str(self.position)] = {"acc": str(accuracy), "loss": str(loss), "visited": 1}
            # with open(self.cache_file, 'w') as outfile:
            #     json.dumps(self.cache, outfile)
            json.dump(self.cache, codecs.open(self.cache_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


        return accuracy, loss

    def step(self, action):
        s = self.position
        last_accuracy = self.current_accuracy
        last_loss = self.current_loss

        base_action = self.action_space[action]
        dimension = base_action[0]
        direction = base_action[1]
        if(self.change_possible(direction, dimension)):
            self.change_position(direction, dimension)
            self.neuro.set_config(self.config)

            self.current_accuracy, self.current_loss = self.cache_train()
            if direction == 'up':
                self.position[dimension] += self.value_sets[dimension]["step"]
            if direction == 'down':
                self.position[dimension] -= self.value_sets[dimension]["step"]
        else:
            print("illegal move. dimension: " + str(dimension) + " move:" + direction)



        s_ = self.position


        # reward function
        if(self.step_count > 0):
            reward = 0
            # if(self.current_accuracy > last_accuracy):
            #     reward += 1
            # elif(self.current_accuracy < last_accuracy):
            #     reward -= 1
            # if(self.current_loss < last_loss):
            #     reward += 0.5
            # elif(self.current_loss > last_loss):
            #     reward -= 0.5
            # if(self.current_accuracy > 0.8):
            #     reward += 1
            # if(self.current_accuracy > 0.85):
            #     reward += 1
            # if(self.current_accuracy < 0.8):
            #     reward -= 1
            # if(self.current_accuracy < 0.75):
            #     reward -= 1
            # if(self.current_accuracy > 0.9):
            #     reward += 1
            reward = (self.current_accuracy - 0.83) * 10
        else:
            reward = 0

        #save results
        with open(self.save_file, 'a') as outfile:
            outfile.write(str(self.step_count) + ";" + str(self.current_accuracy) + ";" + str(self.current_loss) + ";" + str(reward) + ";" + str(dimension) + ";" + direction + ";" + str(self.position) + "\n")

        self.step_count += 1
        if self.step_count == self.steps_per_round:
            done = True
            self.step_count = 0
        else:
            done = False


        return np.array(list(s_.values())), reward, done

    def reduce_possible(self, dimension):
        return self.value_sets[dimension]["min"] <= self.config[dimension] - self.value_sets[dimension]["step"]

    def increase_possible(self, dimension):
        return self.value_sets[dimension]["max"] >= self.config[dimension] + self.value_sets[dimension]["step"]

    def change_possible(self, direction, dimension):
        if direction == 'up':
            return self.increase_possible(dimension)
        if direction == 'down':
            return self.reduce_possible(dimension)

    def reduce_position_value(self, dimension):
        if self.reduce_possible(dimension):
            self.config[dimension] -= self.value_sets[dimension]["step"]

    def increase_position_value(self, dimension):
        if self.increase_possible(dimension):
            self.config[dimension] += self.value_sets[dimension]["step"]

    def change_position(self, direction, dimension):
        if direction == 'up':
            self.increase_position_value(dimension)
        if direction == 'down':
            self.reduce_position_value(dimension)
