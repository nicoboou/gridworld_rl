"""
Policy Iteration algorithm for Gridworld problem.

We are assuming that:
    - we start with the equiprobable policy;
    - when the action send us to a cell outside the grid, we will stay in the same cell.
"""
import numpy as np
from gridworld import Grid_World
import pygame, sys, time
from pygame.locals import *

class PolicyIteration():
    def __init__(self, surface,transition_timestep,board_size,start_coord,original_wall,new_wall,pauseTime, v0_val, gamma, theta, seed):
        """
        Initialize our PolicyIteration class.

        Args:
            surface (): 
            transition_timestep (): 
            board_size (int):
            original_wall (): 
            new_wall (): 
            pauseTime ():
            v0_val (int): initial value for the value function
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process)
            seed (int): seed (for matter of reproducible results)
        """
        self.surface = surface
        self.transition_timestep = transition_timestep
        self.board_size = board_size # int()
        self.start_coord = start_coord
        self.original_wall = original_wall
        self.new_wall = new_wall
        self.pauseTime = pauseTime
        self.v0_val = v0_val
        self.gamma = gamma
        self.theta = theta
        self.seed = seed

        self.v = []
        self.pi = []

    def policy_iteration(self):
        """
        Runs the Policy Iteration algorithm:
            - Policy Evaluation
            - Policy Improvement

        Args:
            board (Environment): gridworld environment
            p_barrier (float): probability of a cell being a barrier
            r_barrier (int): reward for the barrier cells
            v0_val (int): initial value for the value function
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible (i.e. when we can stop process)
            seed (int): seed (for matter of reproducible results)
        """
        # Data storage initialization
        return_mem = []
        timestep_mem = []
        greedy_return_mem = []
        timesteps = 0
        flag = 0

        #Set new wall after certain nb of timesteps + set exploration_epsilon higher
        if timesteps >= self.transition_timestep:
            board = Grid_World(self.surface, self.board_size, self.new_wall,self.start_coord)
            self.epsilon = 0.5
        else:
            board = Grid_World(self.surface, self.board_size, self.original_wall,self.start_coord)

        # Draw objects
        board.draw()

        #Instantiate rewards list
        board.instanciate_rewards_list()

        # Refresh the display
        board.update()
        pygame.display.update()

        # Import board metrics
        board_height = board.board_size[0]
        board_width = board.board_size[1]
        
        # Generate initial value function and policy
        self.v = self.get_init_v(board_height,board_width, self.v0_val, board.goal_coord)
        self.pi = self.get_equiprobable_policy(board_height,board_width,board.actions)

        # Send initial value function and policy to grid
        board.update_value_function(self.v)
        #board.update_optimal_actions(board, self.pi)

        #Initialize policy as a NOT STABLE one
        policy_stable = False

        while not policy_stable:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                    # Handle additional events

            timesteps += 1
            print(f"\nIteration {timesteps} of Policy Iteration algorithm")

            ############ Policy Evaluation Step ############
            self.policy_evaluation(board, self.v, self.pi, self.gamma, self.theta)

            ############ Policy Improvement Step ############
            policy_stable = self.policy_improvement(board, self.v, self.pi, self.gamma)
            #board.update_optimal_actions(board, self.pi)

            if timesteps >= self.transition_timestep and not flag:
                flag = 1
                break

            board.update()

            # Refresh the display
            pygame.display.update()

            # Set the frame speed by pausing between frames
            time.sleep(self.pauseTime)

        print(f"\nPolicy Iteration algorithm converged after {timesteps} steps")


    def policy_evaluation(self,board, v, pi, gamma, theta):
        """
        Applies the policy evaluation algorithm.

        Args:
            board (Environment): gridworld environment
            v (array): numpy array representing the value function
            pi (array): numpy array representing the policy
            gamma (float): gamma parameter (between 0 and 1)
            theta (float): threshold parameter that defines when the change in the value function is negligible
        """

        delta = theta + 1
        iter = 0

        while delta >= theta:
            old_v = v.copy()
            delta = 0

            # Traverse all states
            for x in range(board.board_size[0]):
                for y in range(board.board_size[1]):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(board, v, old_v, board.position, pi, gamma)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - v[x, y]))

            iter += 1

        # Send new value function to grid
        board.update_value_function(v)
        print(f"\nThe Policy Evaluation algorithm converged after {iter} iterations")


    def policy_improvement(self,board, v, pi, gamma):
        """
        Applies the Policy Improvement step.

        Args:
            board (Environment): gridworld environment
            v (array): numpy array representing the value function
            pi (array): numpy array representing the policy
            gamma (float): gamma parameter (between 0 and 1)
        """
        policy_stable = True

        # Iterate states
        for x in range(board.board_size[0]):
            for y in range(board.board_size[1]):
                old_pi = pi[x, y, :].copy()

                # Iterate all actions
                best_actions = []
                max_v = None
                for a in board.actions:
                    # Perform action
                    board.step(a)
                    # Get next state
                    next_state = board.position
                    # Get value
                    curr_val = board.rewards_list[next_state[0], next_state[1]] + gamma * v[next_state[0], next_state[1]]

                    if max_v is None:
                        max_v = curr_val
                        best_actions.append(a)
                    elif curr_val > max_v:
                        max_v = curr_val
                        best_actions = [a]
                    elif curr_val == max_v:
                        best_actions.append(a)

                # Define new policy
                self.define_new_policy(pi, board.position, best_actions, board.actions)

                # Check whether the policy has changed
                if not (old_pi == pi[x,y, :]).all():
                    policy_stable = False

        return policy_stable


    def bellman_update(self,board, v, old_v, current_state, pi, gamma):
        """
        Applies the Bellman update rule to the value function.

        Args:
            board (Environment): grid world environment
            v (array): numpy array representing the value function
            old_v (array): numpy array representing the value function on the last iteration
            current_state (tuple): Current coords (X,Y) of agent on the board (delivered by board.position)
            pi (array): numpy array representing the policy
            gamma (float): gamma parameter (between 0 and 1)
        """

        # The value function on the terminal state always has value 0
        if current_state[0] == board.goal_coord[0] and current_state[1] == board.goal_coord[1]:
            return None

        total = 0

        for a in board.actions:
            # Perform action
            board.step(a)

            # Get next state
            print(board.position)
            next_state_x = board.position[0]
            next_state_y = board.position[1]
            current_state_x = current_state[0]
            current_state_y = current_state[1]

            total += pi[current_state_x, current_state_y, a] * (board.rewards_list[next_state_x, next_state_y] + (gamma * old_v[next_state_x,next_state_x]))
            #print(f"reward:{gamma * old_v[next_state_x,next_state_x]}")

        # Update the value function
        v[current_state[0], current_state[1]] = total


    def define_new_policy(self,pi, current_state, best_actions, actions):
        """
        Defines a new policy given the new best actions (computed by the Policy improvement)

        Args:
            pi (array): numpy array representing the policy
            current_state (tuple): Current coords (X,Y) of agent on the board (delivered by board.position)
            best_actions (list): list with best actions
            actions (list): list of every possible action (given by board.actions)
        """

        prob = 1/len(best_actions)

        for a in actions:
            pi[current_state[0], current_state[1], a] = prob if a in best_actions else 0


    def get_init_v(self,board_height,board_width, v0,goal_coord):
        """
        Defines initial value function v_0.

        Args:
            board_height (int): height of the grid
            board_width (int): width of the grid
            v0 (float): initial value for the value function (equal for every state)
            goal_coord (tuple): Coordinates (X,Y) of the goal/target
        Returns:
            v0 (array): initial value function
        """

        # Init
        v0 = v0 * np.ones((board_height, board_width))

        # ATTENTION: Value function of terminal state must be 0
        v0[goal_coord[0], goal_coord[1]] = 0

        return v0


    def get_equiprobable_policy(self,board_height,board_width,actions):
        """
        Defines the equiprobable policy.
        
        - Policy is a matrix s.t. pi[x, y, a] = Pr[A = a | S = (x,y)]

        - Args:
            board_height (int): height of the grid
            board_width (int): width of the grid
            actions (array): Array of actions delivered by board.actions

        - Returns:
            pi (array): numpy array representing the equiprobably policy
        """
        nb_actions = len(actions)
        pi = 1/nb_actions * np.ones((board_height, board_width, nb_actions))
        return pi


    def send_v_values(self,v):
        """
        Function that sends the array of value functions for each state to grid.

        Args:
            v (array): array of the value functions
            
        """
        pass
        

    def send_optimal_actions(self,board, pi):
        """
        Sends the optimal action to take in each state to the grid.

        Args:
            board (Environment): grid world environment
            pi (array): numpy array indicating the probability of taking each action in each state
        """
        pass


    def get_arrow(self,prob_arr):
        """
        Returns the arrows that represent the highest probability actions.

        Args:
            prob_arr (array): numpy array denoting the probability of taking each action on a given state

        Returns:
            arrow (str): string denoting the most probable action(s)
        """

        best_actions = np.where(prob_arr == np.amax(prob_arr))[0]
        if len(best_actions) == 1:
            if 0 in best_actions:
                return r"$\leftarrow$"
            if 1 in best_actions:
                return r"$\uparrow$"
            if 2 in best_actions:
                return r"$\rightarrow$"
            else:
                return r"$\downarrow$"

        elif len(best_actions) == 2:
            if 0 in best_actions and 1 in best_actions:
                return r"$\leftarrow \uparrow$"
            elif 0 in best_actions and 2 in best_actions:
                return r"$\leftrightarrow$"
            elif 0 in best_actions and 3 in best_actions:
                return r"$\leftarrow \downarrow$"
            elif 1 in best_actions and 2 in best_actions:
                return r"$\uparrow \rightarrow$"
            elif 1 in best_actions and 3 in best_actions:
                return r"$\updownarrow$"
            elif 2 in best_actions and 3 in best_actions:
                return r"$\downarrow \rightarrow$"

        elif len(best_actions) == 3:
            if 0 not in best_actions:
                return r"$\updownarrow \rightarrow$"
            elif 1 not in best_actions:
                return r"$\leftrightarrow \downarrow$"
            elif 2 not in best_actions:
                return r"$\leftarrow \updownarrow$"
            else:
                return r"$\leftrightarrow \uparrow$"

        else:
            return r"$\leftrightarrow \updownarrow$"