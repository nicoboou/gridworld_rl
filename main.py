from argparse import ArgumentParser
from signal import pause
from q_learner import Q_learning
from policy_iteration import PolicyIteration
import pygame
from pygame.locals import *

num_episodes = 200
board_size = [10, 10]
original_wall = [[2, i] for i in range(board_size[1] - 1)]
new_wall = [[2, i] for i in range(1, board_size[1])]
pauseTime = 5  # smaller is faster game
action_dict = {
    "0": "Up",
    "1": "Down",
    "2": "Right",
    "3": "Left",
}
render_env = False
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
    n = board_size[0] * board_size[1]

    args = parser.parse_args()

    if args.type_of_strategy == "qlearning":
        agent = Q_learning(
            alpha=0.5,
            gamma=0.95,
            lmbda=0.0,
            epsilon=0.1,
            n=n,
            num_actions=len(action_dict),
            num_episodes=num_episodes,
            surface = surface,
            board_size=board_size,
            original_wall = original_wall,
            new_wall=new_wall,
            pauseTime=pauseTime,
            action_dict = action_dict,
            render_env=render_env,
            transition_timestep=transition_timestep,
            final_epsilon=final_epsilon,
            anneal_epsilon_episodes=anneal_epsilon_episodes,
            epsilon_anneal_rate=epsilon_anneal_rate
        )
        agent.q_learning()
    
    if args.type_of_strategy == "policy_iter":
        agent = PolicyIteration( surface = surface,transition_timestep = transition_timestep,board_size = board_size,original_wall = original_wall,new_wall=new_wall,pauseTime=pauseTime, v0_val=0, gamma=0.9, theta=0.01, seed=42)
        agent.policy_iteration()
        

    if args.type_of_strategy == "value_iter":
        #agent = Value_Iteration()
        #agent.play()
        pass

