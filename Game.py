from ale_py import ALEInterface
import numpy as np
from Player import Player

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
    
    def __init__(self,display=False,random_seed=123,frame_skip=5,rom_file='qbert.bin'):
        self.ale = ALEInterface()
        
        self.ale.setInt("random_seed", random_seed)
        self.ale.setInt('frame_skip', frame_skip)

        if display:
            import pygame
            pygame.init()
            self.ale.setBool('sound', sound)
            self.ale.setBool('display_screen', display)
        
        # Load the ROM file
        self.ale.loadROM(rom_file)
        
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
        self.disc_states = [1, 1]
        
    def update(self):
        self.ale.getScreenRGB(self.screen)
        self.update_goal_col()
        self.update_disc_states()
        self.player.lives = self.ale.lives()

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
                    pass
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
                    self.player.pos = (row_num, block_num)

                block_num += 1
            row_num += 1

    def update_goal_col(self):
        raw = self.screen[5:30, 30:40]
        raw = raw[np.nonzero(raw)]
        if len(raw > 2):
            self.goal_col = [raw[0], raw[1], raw[2]]

    def update_disc_states(self):
        bl = self.COLOUR["bl"]
        # Checking if disc pixels are visible
        left_disc = not (bl == self.screen[self.DISC_POS[0][1]][self.DISC_POS[0][0]]).all()
        right_disc = not (bl == self.screen[self.DISC_POS[1][1]][self.DISC_POS[1][0]]).all()
        self.disc_states = [left_disc, right_disc]
                
    def is_over(self):
        return self.ale.game_over()
    
    def reset(self):
        return self.ale.reset_game()
