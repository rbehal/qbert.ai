from ale_py import ALEInterface
import numpy as np
from Player import Player
from copy import deepcopy

class Game:
    # (x, y) coordinates for the different blocks 
    BLOCK_POS = [[(76,35)], 
                [(64,63),(92,63)],
                [(53,92),(77,92),(104,92)],
                [(40,121),(64,121),(92,121),(117,121)],
                [(29,150),(52,150),(76,150),(105,150),(128,150)],
                [(16,179),(40,179),(64,179),(93,179),(116,179),(140,179)]]
    # RGB colour values
    COLOUR = { "b": [45, 87, 176],
               "y": [210, 210, 64],
               "g": [50, 132, 50],
               "p": [146, 70, 192],
               "q": [181, 83, 40], # Qbert colour
               "bl": [0, 0, 0] }
    # (x, y) offset for searching for entities and enemies 
    POS_OFFSET = (5, 20)
    # (x, y) coordinates of the left and right disk respectively
    DISC_POS = [(15,138), (144,138)]
    
    def __init__(self,display=False,random_seed=123,frame_skip=5,rom_file='qbert.bin',gamestate=None):
        if gamestate is not None:
            self.screen = deepcopy(gamestate.screen)
            self.goal_col = deepcopy(gamestate.goal_col)
            self.block_states = deepcopy(gamestate.block_states)
            self.enemy_states = deepcopy(gamestate.enemy_states)
            self.entity_states = deepcopy(gamestate.entity_states)
            self.disc_states = deepcopy(gamestate.disc_states)
            self.player = Player(None, player=gamestate.player)
            return
        
        # Initialize ALE and settings
        self.ale = ALEInterface()
        
        self.ale.setInt("random_seed", random_seed)
        self.ale.setInt('frame_skip', frame_skip)

        # Display initialization
        if display:
            import pygame
            pygame.init()
            self.ale.setBool('sound', sound)
            self.ale.setBool('display_screen', display)
        
        # Load the ROM file
        self.ale.loadROM(rom_file)

        # Set RAM
        self.RAM_size = self.ale.getRAMSize()
        self.RAM = np.zeros(self.RAM_size, dtype=np.uint8)
        self.ale.getRAM(self.RAM)
        
        # Set height and width, initialize RGB display array
        (h,w) = self.ale.getScreenDims()
        self.screen = np.zeros([h, w, 3], dtype=np.uint8)
        
        self.player = Player(self.ale)

        # Goal states for blocks
        self.goal_col = [0,0,0]
        self.block_states = [[0], 
                            [0,0],
                            [0,0,0],
                            [0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0,0]]
        # States of friendly green entities
        self.entity_states = [[0], 
                            [0,0],
                            [0,0,0],
                            [0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0,0]]
        # States of enemies 
        self.enemy_states = [[0], 
                            [0,0],
                            [0,0,0],
                            [0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0,0]]
        # States of discs
        self.disc_states = [self.DISC_POS[0], self.DISC_POS[1]]

        # Game statistics
        self.sams_killed = 0 
        self.coilys_killed = 0 
        self.green_balls_caught = 0
        # Initialize high scores
        # total game score, # of sams killed, # of coilys killed, # of green balls
        self.high_scores = [0,0,0,0] 

    # Update all relevant game state parameters
    def update(self):        
        self.update_RAM()
        self.ale.getScreenRGB(self.screen)
        self.update_goal_col()
        self.update_disc_states()
        self.player.lives = self.ale.lives()

        # Loop through all of the blocks
        row_num = 0
        for row in self.BLOCK_POS:
            block_num = 0            
            for block_pos in row:
                x,y = block_pos
                POS_OFFSET = self.POS_OFFSET

                # Setting block state based on colour of surface                
                surface_colour = self.screen[y][x]
                if np.all(surface_colour == self.goal_col):
                    self.block_states[row_num][block_num] = 1
                elif np.all(surface_colour == self.COLOUR["q"]):
                    self.block_states[row_num][block_num] = 1
                else:
                    self.block_states[row_num][block_num] = 0

                # Area above block to search for various entities, enemies, and the player 
                search_area = self.screen[y-POS_OFFSET[1]:y,x-POS_OFFSET[0]:x+POS_OFFSET[0]]
                flat_search = search_area.reshape(POS_OFFSET[0]*POS_OFFSET[1]*2,3)
                
                # Checking for enemies (purple)
                contains_enemies = (self.COLOUR["p"] == flat_search).all(1).any()
                if contains_enemies:
                    self.enemy_states[row_num][block_num] = 1
                else:
                    self.enemy_states[row_num][block_num] = 0

                # Checking for entities (green)
                contains_entities = (self.COLOUR["g"] == flat_search).all(1).any()
                if contains_entities:
                    self.entity_states[row_num][block_num] = 1
                else:
                    self.entity_states[row_num][block_num] = 0

                # Checking for qbert (pinkish)
                contains_qbert = (self.COLOUR["q"] == flat_search).all(1).any()
                if contains_qbert:
                    self.player.pos = self.BLOCK_POS[row_num][block_num]

                block_num += 1
            row_num += 1

    # Updates the target colour for the blocks Q*bert changes
    def update_goal_col(self):
        # Check the area of the score for a colour
        raw = self.screen[5:30, 30:40]
        raw = raw[np.nonzero(raw)]
        if len(raw > 2):
            self.goal_col = [raw[0], raw[1], raw[2]]

    def update_disc_states(self):
        bl = self.COLOUR["bl"]
        # Checking if disc pixels are visible
        left_disc = None if (bl == self.screen[self.DISC_POS[0][1]][self.DISC_POS[0][0]]).all() else self.DISC_POS[0]
        right_disc = None if (bl == self.screen[self.DISC_POS[1][1]][self.DISC_POS[1][0]]).all() else self.DISC_POS[1]
        self.disc_states = [left_disc, right_disc]

    def update_RAM(self):
        self.ale.getRAM(self.RAM)

    # Gets the RGB (x,y ) coordinate values from the internal state variables
    def get_coords_from_state(self, states):
        coords = []
        for i in range(len(states)):
            for j in range(len(states[i])):
                if states[i][j] == 1:
                    coords.append(self.BLOCK_POS[i][j])
        return coords

    # Executes an action making reflective changes in the internal game state only 
    def execute_action(self, action):
        # Get player position in terms of indices 
        player_pos = [(index, row.index(self.player.pos)) for index, row in enumerate(self.BLOCK_POS) if self.player.pos in row]
        if len(player_pos) == 0:
            return

        y,x = player_pos[0]

        if action == "UP":
            new_y = y - 1
            # Checks if move is within bounds
            if new_y < 0:
                self.player.died()
                return
            if x >= len(self.BLOCK_POS[new_y]):
                # Check if player is able to jump to disc
                if new_y == 3 and self.disc_states[1] is not None:
                    self.player.pos = self.disc_states[1]
                    return
                else:
                    self.player.died()
                    return
            self.player.pos = self.BLOCK_POS[new_y][x]
        elif action == "DOWN":
            new_y = y + 1
            # Check if move is within bounds
            if new_y < len(self.BLOCK_POS):
                self.player.pos = self.BLOCK_POS[new_y][x]
            else:
                self.player.died()
                return
        elif action == "RIGHT":
            new_x = x + 1
            new_y = y + 1
            # Check if move is within bounds
            if new_y < len(self.BLOCK_POS):
                self.player.pos = self.BLOCK_POS[new_y][new_x]
            else:
                self.player.died()
                return
        elif action == "LEFT": 
            new_x = x - 1
            new_y = y - 1
            # Check if move is within bounds
            if new_y < 0:
                self.player.died()
                return
            if new_x < 0:
                # Check if player is able to jump to disc
                if new_y == 3 and self.disc_states[0] is not None:
                    self.player.pos = self.disc_states[0]
                    return
                else:
                    self.player.died()
                    return    
            self.player.pos = self.BLOCK_POS[new_y][new_x]   

        # Check if Qbert hit an enemy
        coords = self.get_coords_from_state(self.enemy_states)
        for coord in coords:
            if self.player.pos == coord:
                self.player.died()
                return
        return  

    # Function that is used to get all rewards after agent makes an action 
    # as well as stall until agent is ready again
    def get_reward(self, reward):
        total_reward = 0 
        # Update rewards until agent is ready to move again
        while (not (self.RAM[0] == 2 and self.RAM[self.RAM_size-1] & 1)) or (self.RAM[self.RAM_size - 2] == 41):
            if (self.ale.lives() == 0):
                break           
           
           # Keep track of statistics based on reward received
            total_reward += reward
            if reward == 300:
                self.sams_killed += 1
            elif reward == 500:
                self.coilys_killed += 1
            elif reward == 100:
                self.green_balls_caught += 1

            self.update_RAM()
            reward = self.ale.act(0) # No-Op action to wait for rewards/stall
        
        return total_reward

    # Stalls the game until Q*bert is ready to make a move 
    def initialize(self):
        while not (self.RAM[0] == 2 and self.RAM[self.RAM_size - 1] & 1):  # First byte = 2, Last bit = 1
            self.ale.getRAM(self.RAM)
            self.ale.act(0) # No-Op action to stall until player is ready
        self.update()
            
    # Returns a boolean representing then game status
    def is_over(self):
        return self.ale.game_over()
    
    # Resets the game at the end of an episode
    def reset(self, total_reward):
        # Update high scores
        if total_reward > self.high_scores[0]:
            self.high_scores[0] = total_reward
        if self.sams_killed > self.high_scores[1]:
            self.high_scores[1] = self.sams_killed
        if self.coilys_killed > self.high_scores[2]:
            self.high_scores[2] = self.coilys_killed
        if self.green_balls_caught > self.high_scores[3]:
            self.high_scores[3] = self.green_balls_caught

        # Reset statistics 
        self.sams_killed = 0 
        self.coilys_killed = 0 
        self.green_balls_caught = 0

        self.ale.reset_game()
        return 
