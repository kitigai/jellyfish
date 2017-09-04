import sys
import os.path
from ballenv import ballenv

import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras import backend as K

MODEL_FILE_NAME = "./save_model/jellyfish_dqn.h5"

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False
        
        # get size of state and action
        self.state_size = (state_size[0], state_size[1], 4)
        print(self.state_size)
        self.action_size = action_size
        
        # hyper parameters for DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        
        # create replay memory using deque
        self.memory = deque(maxlen=400000)
        
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        # initialize target model
        self.update_target_model()
        
        # define optimizer
        self.optimizer = self.optimizer()
        
        
        if os.path.isfile(MODEL_FILE_NAME):
            self.model.load_weights(MODEL_FILE_NAME)
            
    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train
      
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        update_target = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        loss = self.optimizer([update_input, action, target])
    
if __name__ == "__main__":
    env = ballenv()
    
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    
    scores, episodes = [], []
    
    done = False
    score = 0
    state = env.reset()
    step = 0
    history = np.stack((state,state,state,state), axis=2)
    history = np.reshape([history],(1, 65, 65, 4))
    try:
        while 1:

            # get action for the current state and go one step in environment
            action = agent.get_action(history)
            next_state, reward, done = env.step(action)

            # save the sample <s, a, r, s'> to the replay memory
            next_state = np.reshape([next_state], (1, 65, 65, 1))
            next_history = np.append(next_state, history[:,:,:,:3], axis=3)

            agent.append_sample(history, action, reward, next_history, done)
            agent.train_model()
            score += reward
            history = next_history
            step = step + 1
            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()
                print("step:", step, "score:", score)

    finally:
        agent.model.save_weights(MODEL_FILE_NAME)
