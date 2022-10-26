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
    def __init__(self, surface,transition_timestep,board_size,start_coord,goal_coord,original_wall,new_wall,reward_goal,reward_wall,reward_empty,pauseTime, v0_val, gamma, theta, seed):
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
        self.goal_coord = goal_coord
        self.original_wall = original_wall
        self.new_wall = new_wall
        self.reward_goal = reward_goal
        self.reward_wall = reward_wall
        self.reward_empty = reward_empty
        self.pauseTime = pauseTime
        self.v0_val = v0_val
        self.gamma = gamma
        self.theta = theta
        self.seed = seed

        self.v = []
        self.pi = []
        self.optimal_actions = []

    def policy_iteration(self):
        """
        Runs the Policy Iteration algorithm:
            - Policy Evaluation
            - Policy Improvement

        Args:
            board (Environment): gridworld environment
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
            board = Grid_World(self.surface, self.board_size,self.new_wall,self.start_coord,self.goal_coord,self.reward_goal,self.reward_wall,self.reward_empty)
            self.epsilon = 0.5
        else:
            board = Grid_World(self.surface, self.board_size, self.original_wall,self.start_coord,self.goal_coord,self.reward_goal,self.reward_wall,self.reward_empty)

        # Draw objects
        board.draw()

        #Instantiate rewards list
        board.instanciate_rewards_list()

        # Import board metrics
        board_height = board.board_size[0]
        board_width = board.board_size[1]
        
        # Generate initial value function and policy
        self.v = self.get_init_v(board_height,board_width, self.v0_val, board.goal_coord)
        self.pi,self.optimal_actions = self.get_equiprobable_policy(board_height,board_width,board.actions)

        # Send initial value function and policy to grid
        board.update_value_function(self.v)
        board.update_optimal_actions(self.optimal_actions)
        

        # Refresh the display
        board.update()
        pygame.display.update()

        #Initialize policy as a NOT STABLE one
        policy_stable = False
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                # Handle additional events
        
            while(True):
                if not policy_stable:
                    timesteps += 1
                    print(f"\nIteration {timesteps} of Policy Iteration algorithm")

                    ############ Policy Evaluation Step ############
                    self.policy_evaluation(board, self.v, self.pi, self.gamma, self.theta)

                    # Set the frame speed by pausing between frames

                    ############ Policy Improvement Step ############
                    policy_stable = self.policy_improvement(board, self.v, self.pi, self.gamma)

                    # Set the frame speed by pausing between frames

                    if timesteps >= self.transition_timestep and not flag:
                        flag = 1
                        break

                    board.update()

                    # Refresh the display
                    pygame.display.update()

                    print(f"\The whole Policy Iteration (eval -> improvement -> eval -> ...) algorithm converged after {timesteps} steps")

                else:
                    time.sleep(5)
                    
                    

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
            for x in range(board.board_size[0]): #[0,...,9]
                for y in range(board.board_size[1]): #[0,...,9]
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(board, v, old_v, x, y, pi, gamma)
                    # Compute difference for EACH STATE, and take the maximum difference
                    delta = max(delta, abs(old_v[x, y] - v[x, y]))

            # Send new value function to grid
            board.update_value_function(v)
            time.sleep(self.pauseTime)
            board.draw()
            pygame.display.update()
            
            iter += 1

        print(f"\nValue function updated: the Policy Evaluation algorithm converged after {iter} sweeps")


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
        for x in range(board.board_size[0]-1):
            for y in range(board.board_size[1]-1):
                old_pi = pi[x, y, :].copy()

                # Instanciate best actions list & best action val
                best_actions = []
                max_action_val = None

                ############ COMPUTE the ACTION-value function Q_𝜋(s,a) for each action ############
                for a in board.actions:
                    # Get next state
                    board_height = board.board_size[0]
                    board_length = board.board_size[1]
                    s_prime_x, s_prime_y = self.get_next_state(x,y,a,board_height,board_length)  

                    # Get ACTION value 
                    curr_action_value = board.rewards_list[s_prime_x, s_prime_y] + gamma * v[s_prime_x, s_prime_y]

                    if max_action_val is None: #If no best action, add this one
                        max_action_val = curr_action_value
                        best_actions.append(a)
                    elif curr_action_value > max_action_val: #If better than precedent action, replace
                        max_action_val = curr_action_value
                        best_actions = [a]
                    elif curr_action_value == max_action_val: # If the Action-value for this specific actions equals another Action-value of another action, both deserve to be taken
                        best_actions.append(a)

                # Define new policy π(a|s) with the following variables
                # - pi: current policy, will be updated by new_policy
                # - current_state: state for which the policy should be updated
                # - new_policy: best actions to take for this specific state
                # - actions: all set of actions 
                self.pi[x,y] = self.improve_policy(pi, board.position, best_actions, board.actions)
        
                # Get arrows for Best Actions for this specific state (x,y)
                self.optimal_actions[x,y] = self.get_arrow(self.pi[x, y, :]) 



                # Check whether the policy has changed
                if (old_pi == self.pi[x,y, :]).all():
                    policy_stable = True

        if not policy_stable:
            # Update arrows on grid
            board.update_optimal_actions(self.optimal_actions)
            # Refresh the display
            board.update()
            pygame.display.update()
            print(f"\nPolicy improved for all states.")
        else:
            # Update arrows on grid
            board.update_optimal_actions(self.optimal_actions)
            # Refresh the display
            board.update()
            pygame.display.update()
            print(f"\nPolicy is now STABLE !")
        return policy_stable


    def bellman_update(self,board, v, old_v, x,y, pi, gamma):
        """
        Applies the Bellman update rule to the value function.

        Args:
            board (Environment): grid world environment
            v (array): numpy array representing the value function
            old_v (array): numpy array representing the value function on the last iteration
            x (int): x coord of current state
            y (int): y coord of current state
            gamma (float): gamma parameter (between 0 and 1)
        """

        # The value function on the terminal state always has value 0
        if x == board.goal_coord[0] and y == board.goal_coord[1]:
            return None

        total = 0

        for a in board.actions:

            # Get next state
            board_height = board.board_size[0]
            board_length = board.board_size[1]
            s_prime_x, s_prime_y = self.get_next_state(x,y,a,board_height,board_length)  

            total += pi[x, y, a] * (board.rewards_list[s_prime_x, s_prime_y] + (gamma * old_v[s_prime_x,s_prime_y]))

        # UPDATE OF the VALUE function
        v[x, y] = total


    def improve_policy(self,pi, current_state, best_actions, actions):
        """
        Defines a new policy π(a|s) given the new best actions (computed by the Policy improvement)

        Args:
            pi (array): numpy array representing the policy
            current_state (tuple): Current coords (X,Y) of agent on the board (delivered by board.position)
            best_actions (list): list with best actions
            actions (list): list of every possible action (given by board.actions)
        """

        prob = 1/len(best_actions)
        
        for a in actions:
            pi[current_state[0], current_state[1], a] = prob if a in best_actions else 0
        
        return pi[current_state[0], current_state[1]]


    def get_next_state(self,x,y,action,board_height,board_width):
        """Computes next state from current state and action.
        Args:
            x (int): x value of the current state
            y (int): y value of the current state
            a (int): action
            n (int): length and width of the grid
        Returns:
            s_prime_x (int): x value of the next state
            s_prime_y (int): y value of the next state
        """

        # Compute next state according to the action

    
        if action == 0:  # Action Up
                s_prime_x = max(0,x - 1) #forbids to go out of board
                s_prime_y = y

        elif action == 1:  # Action Down
                s_prime_x = min(board_height -1 ,x + 1) #forbids to go out of board
                s_prime_y = y

        elif action == 2:  # Action Right
                s_prime_x = x
                s_prime_y = min(board_width-1,y + 1) #forbids to go out of board

        elif action == 3:  # Action Left
                s_prime_x = x
                s_prime_y = max(0,y - 1) #forbids to go out of board

        return s_prime_x, s_prime_y

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
        pi = 1/nb_actions * np.ones((board_height, board_width,nb_actions)) #One policy per action, for each state => p[x,y] = [0.25,0.25,0.25,0.25] & p[x,y,a] = 0.25
        opt_act_temp = []
        for x in range(board_height):
            opt_act_temp.append("all_arrows")
            x += 1
            for y in range(board_width):
                self.optimal_actions.append(opt_act_temp)
                y += 1

        self.optimal_actions = np.array(self.optimal_actions)
        
        return pi, self.optimal_actions


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
            if 0 in best_actions: #Action Up is best
                return "up_arrow"
            elif 1 in best_actions: #Action Down is best
                return "down_arrow" 
            elif 2 in best_actions:
                return "right_arrow" #Action Right is best
            else:
                return "left_arrow" #Action Left is best

        elif len(best_actions) == 2:
            if 0 in best_actions and 1 in best_actions:
                return "up_down_arrow"
            elif 0 in best_actions and 2 in best_actions:
                return "up_right_arrow"
            elif 0 in best_actions and 3 in best_actions:
                return "up_left_arrow"
            elif 1 in best_actions and 2 in best_actions:
                return "down_right_arrow"
            elif 1 in best_actions and 3 in best_actions:
                return "left_down_arrow"
            elif 2 in best_actions and 3 in best_actions:
                return "left_right_arrow"

        elif len(best_actions) == 3:
            if 0 not in best_actions:
                return "down_left_right_arrow"
            elif 1 not in best_actions:
                return "left_right_up_arrow"
            elif 2 not in best_actions:
                return "up_down_left_arrow"
            else:
                return "up_down_right_arrow"
        else:
            return "all_arrows"