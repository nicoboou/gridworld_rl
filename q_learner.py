import numpy as np
from gridworld import Grid_World
import pygame, sys, time
from pygame.locals import *

class Q_learning():
    def __init__(self, alpha = 0.1, gamma = 0.99, lmbda=0.1, epsilon = 0.1, n = 54, num_actions = 4, num_episodes = 200,surface= (600,600), board_size = [10,10],original_wall = [],new_wall=[],pauseTime=0.1,render_env=False,transition_timestep=1000,final_epsilon=0.01,anneal_epsilon_episodes=10,epsilon_anneal_rate=0):
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.n = n
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.surface = surface
        self.board_size = board_size
        self.original_wall = original_wall
        self.new_wall = new_wall
        self.pauseTime = pauseTime  # smaller is faster game
        self.render_env = render_env
        self.transition_timestep = transition_timestep
        self.final_epsilon = final_epsilon
        self.anneal_epsilon_episodes = anneal_epsilon_episodes
        self.epsilon_anneal_rate = epsilon_anneal_rate

        self.w = np.zeros((n,num_actions))
        self.e = np.zeros((n,num_actions))
        self.delta = 0.0
        self.q_value = 0.0
        self.next_q_value = 0.0
        self.greedy = False


    def Q_value(self, features, action):
        self.q_value = self.w[features,action]

    def calc_q_value(self, features, action):
        return self.w[features,action]

    def greedy_Q(self, features):
        q_vals = self.w[features,:]
        maxQ = np.max(q_vals)

        # In the 1st epsiode, all actions can have the same Q-value. So a random action needs to be chosen
        if np.unique(q_vals).size == 1:
            greedy_action = np.random.choice(range(self.num_actions))
        else:
            greedy_action = np.argmax(q_vals)
        return  maxQ, greedy_action

    def Next_Q_value(self,features):
        q_vals = self.w[features, :]
        self.next_q_value = np.max(q_vals)

    def calc_delta(self, reward):
        self.delta = reward + self.gamma*self.next_q_value - self.q_value

    def weight_update(self, features, action):
        self.w[features, action] += self.alpha*self.delta

    def sample_action(self, features,nb_actions):
        maxQ, greedy_action = self.greedy_Q(features=features)
        if np.random.rand() < self.epsilon:
            #action = np.random.choice(range(self.num_actions))
            action = np.random.choice(nb_actions)
            self.greedy = False
        else:
            action = greedy_action
            self.greedy = True
        return action

    def async_calc_delta(self, reward):
        return reward + self.gamma * self.next_q_value - self.q_value

    def master_func(self, current_features, next_features, reward, action):
        self.Q_value(current_features, action)
        self.Next_Q_value(next_features)
        self.calc_delta(reward)
        self.weight_update(features=current_features, action=action)

    def get_features(self,pos):
        return pos[0] * (self.board_size[1] - 1) + pos[1]

    def q_learning(self):
        # Data storage initialization
        return_mem = []
        timestep_mem = []
        greedy_return_mem = []
        timesteps = 0
        flag = 0

        # Loop forever
        for i_episode in range(self.num_episodes):
            # create and initialize objects
            gameOver = False
            #Set new wall after certain nb of timesteps + set exploration_epsilon higher
            if timesteps >= self.transition_timestep:
                board = Grid_World(self.surface, self.board_size, self.new_wall)
                self.epsilon = 0.5
            else:
                board = Grid_World(self.surface, self.board_size, self.original_wall)

            # Draw objects
            board.draw()

            # Refresh the display
            pygame.display.update()

            # Q learner specific initializations
            current_state = board.position
            current_features = self.get_features(board.position)
            episode_return = 0
            episode_timesteps = 0

            while not gameOver:
                # Handle events
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                        # Handle additional events
                # Choose and execute an action
                action = self.sample_action(current_features,board.actions)
                board.step(action)

                # Transition to next state
                next_state = board.position
                next_features = self.get_features(next_state)

                # Q-learning update
                self.master_func(current_features, next_features, board.reward_qlearning, action)

                current_features = next_features

                episode_return += board.reward_qlearning
                episode_timesteps += 1
                timesteps += 1

                if timesteps >= self.transition_timestep and not flag: #If timesteps are over a certain milestone,
                    flag = 1
                    break

                # print "Board position = ", board.position, " Action = ", action_dict[str(action)],\
                #    "Q-value = ", self.q_value, "TD Error = ", self.delta, "Timesteps = ", episode_timesteps

                # Update and draw objects for next frame
                gameOver = board.update()

                # Refresh the display
                pygame.display.update()

                # Set the frame speed by pausing between frames
                time.sleep(self.pauseTime)
            print(
                "Episode ",
                i_episode + 1,
                " ended in ",
                episode_timesteps,
                " timesteps and return = ",
                episode_return,
                "Total Timesteps = ",
                timesteps,
            )
            return_mem.append(episode_return)
            timestep_mem.append(episode_timesteps)
            # eval_return, eval_time = eval_policy(self, surface)
            # greedy_return_mem.append([eval_return, eval_time])
        np.savetxt("Episode_returns", return_mem)
        np.savetxt("Episode_time", timestep_mem)
        np.savetxt("weights_q_learner", self.w)
