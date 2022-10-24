# Grid_World

import pygame
import sys, time, random
from pygame.locals import *
import numpy as np

action_dict = {
    "0": "Up",
    "1": "Down",
    "2": "Right",
    "3": "Left",
}

# User-defined classes


class Tile:
    # An object in this class represents a single Tile that
    # has an image

    # initialize the class attributes that are common to all
    # tiles.

    borderColor = pygame.Color("black")
    borderWidth = 1  # the pixel width of the tile border
    image = pygame.image.load("satellite.png")

    def __init__(self, x, y, wall, surface,value_function_nb,policy_arrow,reward, tile_size=(60, 60)):
        # Initialize a tile to contain an image
        # - x is the int x coord of the upper left corner
        # - y is the int y coord of the upper left corner
        # - image is the pygame.Surface to display as the
        # exposed image
        # - surface is the window's pygame.Surface object

        self.wall = wall
        self.origin = (x, y)
        self.tile_coord = [x // 60, y // 60]
        self.surface = surface
        self.tile_size = tile_size
        self.value_function_nb = value_function_nb
        self.policy_arrow = policy_arrow
        self.reward = reward

    def draw(self, pos, goal,value_function,policy):
        # Draw the tile.

        rectangle = pygame.Rect(self.origin, self.tile_size)
        if self.wall:
            pygame.draw.rect(self.surface, pygame.Color("gray"), rectangle, 0)
        elif goal == self.tile_coord:
            pygame.draw.rect(self.surface, pygame.Color("green"), rectangle, 0)
        else:
            pygame.draw.rect(self.surface, pygame.Color("white"), rectangle, 0)

        if pos == self.tile_coord:
            self.surface.blit(Tile.image, self.origin)

        font  = pygame.font.SysFont( None, 20 )                # Default font, Size 20
        font_reward  = pygame.font.SysFont( None, 17 )                # Default font, Size 20
        
        # Color scale for Value function
        # pct_diff = 0.0 + np.log(abs(round(self.value_function_nb,2)))
        # red_color = min(255, pct_diff*2 * 255)
        # green_color = min(255, round(self.value_function_nb,2)*2 * 255)
        # col = (red_color, green_color, 0)
        value_function_image = font.render(str(round(self.value_function_nb,2)), True, pygame.Color("black"))  # Number assigned as Value function
        policy_arrow_image = font.render(self.policy_arrow,True,pygame.Color("black"), pygame.Color("white"))
        if self.reward > 0:
            reward_image = font_reward.render(str(round(self.reward,1)),True,pygame.Color("blue"))
        elif self.reward < 0:
            reward_image = font_reward.render(str(round(self.reward,1)),True,pygame.Color("red"))
        else:
            reward_image = font_reward.render(str(round(self.reward,1)),True,pygame.Color("black"))

        # centre the VALUE FUNCTION image in the cell by calculating the margin-distance
        margin_x_value = ( self.tile_size[0]-1 - value_function_image.get_width() ) // 2
        margin_y_value = ( self.tile_size[1]-1 - value_function_image.get_height() ) // 2

        # centre the POLICY image in the cell by calculating the margin-distance
        margin_x_policy = ( self.tile_size[0]-1 - policy_arrow_image.get_width() ) // 2
        margin_y_policy = ( self.tile_size[1]-1 - policy_arrow_image.get_height()) // 2

        # set the REWARD image down in the cell by calculating the margin-distance
        margin_x_reward = ( self.tile_size[0]-1 - reward_image.get_width() ) // 2
        margin_y_reward = ( self.tile_size[1]-1 - reward_image.get_height()) // 2

        self.surface.blit(value_function_image,( self.origin[0]+2 + margin_x_value, self.origin[1]+2 + margin_y_value ))
        self.surface.blit(policy_arrow_image,( self.origin[0]+2 + margin_x_policy, self.origin[1]+2 + margin_y_policy ))
        self.surface.blit(reward_image,( self.origin[0] + margin_x_policy, self.origin[1]+25 + margin_y_policy ))

        pygame.draw.rect(self.surface, Tile.borderColor, rectangle, Tile.borderWidth)


class Grid_World:
    # An object in this class represents a Grid_World game.
    tile_width = 60
    tile_height = 60

    def __init__(
        self,
        surface,
        board_size=(10, 10),
        wall_coords=[],
        start_coord=(0, 3),
        goal_coord=(5, 8),
        reward_goal = 1,
        reward_wall = -1,
        reward_empty = 0
    ):
        # Intialize a Grid_World game.
        # - surface is the pygame.Surface of the window

        self.surface = surface
        self.bgColor = pygame.Color("black")
        self.board_size = list(board_size)
        if not wall_coords:
            self.wall_coords = [[2, i] for i in range(board_size[1] - 1)]
        else:
            self.wall_coords = wall_coords

        self.start_coord = list(start_coord)
        self.goal_coord = list(goal_coord)
        self.position = list(start_coord)
        self.actions = range(4)
        self.rewards_list = []
        self.reward_qlearning = 0
        self.reward_goal = reward_goal
        self.reward_wall =  reward_wall
        self.reward_empty = reward_empty

        self.calc_wall_coords()
        self.createTiles()

    def calc_wall_coords(self):
        self.board_wall_coords = [
            [x, y] for x, y in self.wall_coords
         ]

    def instanciate_rewards_list(self):
        self.rewards_list = (self.reward_empty) * np.ones((self.board_size[0], self.board_size[1]))
        self.rewards_list[self.goal_coord[0],self.goal_coord[1]] = self.reward_goal
        for i in self.wall_coords:
            self.rewards_list[i[0],i[1]] = self.reward_wall
        for x,row in enumerate(self.board):
                for y,tile in enumerate(row):
                    tile.reward =  self.rewards_list[x,y]

    def find_board_coords(self, pos):
        x = pos[0]
        #y = self.board_size[0] - pos[0] - 1
        y = pos[1] - 1
        return [x, y]

    def createTiles(self):
        # Create the Tiles
        # - self is the Grid_World game
        self.board = []
        for rowIndex in range(0, self.board_size[0]):
            row = []
            for columnIndex in range(0, self.board_size[1]):
                imageIndex = rowIndex * self.board_size[1] + columnIndex
                x = columnIndex * Grid_World.tile_width
                y = rowIndex * Grid_World.tile_height
                if [rowIndex, columnIndex] in self.board_wall_coords:
                    wall = True
                else:
                    wall = False
                tile = Tile(x, y, wall, self.surface,0,"",0)
                row.append(tile)
            self.board.append(row)

    def update_value_function(self,value_function_array):
        for x,row in enumerate(self.board):
                for y,tile in enumerate(row):
                    tile.value_function_nb =  value_function_array[x,y]

    # def update_policy(self):

    def draw(self):
        # Draw the tiles.
        # - self is the Grid_World game
        pos = self.find_board_coords(self.position)
        goal = self.find_board_coords(self.goal_coord)
        self.surface.fill(self.bgColor)
        for row in self.board:
            for tile in row:
                tile.draw(pos, goal,tile.value_function_nb,tile.policy_arrow)

    def update(self):
        # Check if the game is over. If so return True.
        # If the game is not over,  draw the board
        # and return False.
        # - self is the TTT game

        if self.position == self.goal_coord:
            return True
        else:
            self.draw()
            return False

    def step(self, action):
        x, y = self.position
        if action == 0:  # Action Up
            # print "Up"
            if [x - 1, y] not in self.wall_coords and x - 1 > 0:
                self.position = [x - 1, y]

        elif action == 1:  # Action Down
            # print "Down"
            if [x + 1, y] not in self.wall_coords and x + 1 < self.board_size[0]:
                self.position = [x + 1, y]

        elif action == 2:  # Action Right
            # print "Right"
            if [x, y + 1] not in self.wall_coords and y + 1 < self.board_size[1]:
                self.position = [x, y + 1]

        elif action == 3:  # Action Left
            # print "Left"
            if [x, y - 1] not in self.wall_coords and y - 1 > 0:
                self.position = [x, y - 1]

        # Reward definition
        if self.position == self.goal_coord:
            self.reward_qlearning = 1
        else:
            self.reward_qlearning = 0

    def change_the_wall(self, wall_coords):
        self.wall_coords = wall_coords
        self.calc_wall_coords()
        self.createTiles()

    def change_the_goal(self, goal):
        self.goal_coord = list(goal)


if __name__ == "__main__":
    # Initialize pygame
    pygame.init()

    # Set window size and title, and frame delay
    surfaceSize = (600, 600)
    windowTitle = "Grid_World"
    pauseTime = 1  # smaller is faster game

    # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)
    pygame.display.set_caption(windowTitle)

    # create and initialize objects
    gameOver = False
    board = Grid_World(surface)

    # Draw objects
    board.draw()

    # Refresh the display
    pygame.display.update()

    # Loop forever
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                # Handle additional events

        # Update and draw objects for next frame
        gameOver = board.update()
        if gameOver:
            break

        # Refresh the display
        pygame.display.update()

        #Set the frame speed by pausing between frames
        time.sleep(pauseTime)
