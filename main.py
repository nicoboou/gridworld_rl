from argparse import ArgumentParser
from signal import pause
from q_learner import Q_learning
from policy_iteration import PolicyIteration
import pygame
from pygame.locals import *

######## VARIABLES ########

# GLOBAL VARIABLES
START_COORD = (0,0)
GOAL_COORD  = (5,5)
BOARD_SIZE = [10, 10]
ORIGINAL_WALL = [[2, i] for i in range(BOARD_SIZE[1] - 1)]
NEW_WALL = [[2, i] for i in range(1, BOARD_SIZE[1])]
PAUSE_TIME = 0.01  # smaller is faster game
ACTION_DICT = {
    "0": "Up",
    "1": "Down",
    "2": "Right",
    "3": "Left",
}

# SPECIFIC POLICY ITERATION
REWARD_GOAL = 10
REWARD_WALL = -10
REWARD_EMPTY = 0
V0_VAL=0                    # Initial value of the value functions for each state
GAMMA=0.9                   #Discount factor
THETA=0.01                #threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process of Policy Evaluation)
SEED=42                     #seed (for matter of reproducible results)
render_env = False

# SPECIFIC QLEARNING
NUM_EPISODES = 200
transition_timestep = 3000
final_epsilon = 0.01
anneal_epsilon_episodes = 10
epsilon_anneal_rate = (1.0 - final_epsilon) / float(anneal_epsilon_episodes)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument(
        dest="type_of_strategy",
        default="qlearning",
        type=str,
        help="Choice of strategy.",
    )

    # Init pygame
    pygame.init()

    # Set window size and title, and frame delay
    surfaceSize = (600, 600)
    windowTitle = "Grid_World"

    # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)
    pygame.display.set_caption(windowTitle)

    #Nb of tiles
    n = BOARD_SIZE[0] * BOARD_SIZE[1]

    args = parser.parse_args()

    if args.type_of_strategy == "qlearning":
        agent = Q_learning(
            alpha=0.5,
            gamma=0.95,
            lmbda=0.0,
            epsilon=0.1,
            n=n,
            num_actions= len(ACTION_DICT),
            num_episodes=NUM_EPISODES,
            surface = surface,
            board_size=BOARD_SIZE,
            start_coord=START_COORD,
            original_wall = ORIGINAL_WALL,
            new_wall=NEW_WALL,
            pauseTime=PAUSE_TIME,
            render_env=render_env,
            transition_timestep=transition_timestep,
            final_epsilon=final_epsilon,
            anneal_epsilon_episodes=anneal_epsilon_episodes,
            epsilon_anneal_rate=epsilon_anneal_rate
        )
        agent.q_learning()
    
    if args.type_of_strategy == "policy_iter":
        agent = PolicyIteration( surface = surface,transition_timestep = transition_timestep,board_size = BOARD_SIZE,original_wall = ORIGINAL_WALL,new_wall=NEW_WALL,pauseTime=PAUSE_TIME,start_coord=START_COORD,goal_coord = GOAL_COORD,reward_goal = REWARD_GOAL,reward_wall=REWARD_WALL,reward_empty=REWARD_EMPTY, v0_val=V0_VAL, gamma=GAMMA, theta=THETA, seed=SEED)
        agent.policy_iteration()
        

    if args.type_of_strategy == "value_iter":
        #agent = Value_Iteration()
        #agent.play()
        pass

