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
        self.update_block_states()
        self.update_enemy_states()
        self.update_entity_states()
        self.update_disc_states()
        self.update_player_state()

    def update_goal_col(self):
        raw = self.screen[5:30, 30:40]
        raw = raw[np.nonzero(raw)]
        if len(raw > 2):
            self.goal_col = [raw[0], raw[1], raw[2]]
            
    def update_block_states(self):
        row_num = 0
        for row in self.BLOCK_POS:
            block_num = 0            
            for block_pos in row:
                colour = self.screen[block_pos[1]][block_pos[0]]
                if np.all(colour == self.goal_col):
                    self.block_states[row_num][block_num] = 1
                elif np.all(colour == self.COLOUR["q"]):
                    pass
                else:
                    self.block_states[row_num][block_num] = 0
                block_num += 1
            row_num += 1

    def update_enemy_states(self):
        enemy_blocks = self.search_for_colour(self.COLOUR["p"])
        for block_pos in enemy_blocks:
            self.enemy_states[block_pos[0]][block_pos[1]] = 1

    def update_entity_states(self):
        entity_blocks = self.search_for_colour(self.COLOUR["g"])
        for block_pos in entity_blocks:
            self.entity_states[block_pos[0]][block_pos[1]] = 1

    def update_disc_states(self):
        bl = self.COLOUR["bl"]
        # Checking if disc pixels are visible
        left_disc = not (bl == self.screen[self.DISC_POS[0][1]][self.DISC_POS[0][0]]).all()
        right_disc = not (bl == self.screen[self.DISC_POS[1][1]][self.DISC_POS[1][0]]).all()
        self.disc_states = [left_disc, right_disc]

    def update_player_state(self):
        qbert_block = self.search_for_colour(self.COLOUR["q"], True)
        if (len(qbert_block) == 0):
            self.player.pos = (0, 0)
        else:
            self.player.pos = qbert_block[0]
        self.player.lives = self.ale.lives()

    # Searches for which blocks contain a given colour above/on them
    # Returns coordinates of aforementioned blocks
    def search_for_colour(self,colour,qbert_search=False):
        POS_OFFSET = self.POS_OFFSET
        blocks_with_colour = []

        row_num = 0
        for row in self.BLOCK_POS:
            block_num = 0
            for block_pos in row:
                x,y = block_pos
                
                search_area = self.screen[y-POS_OFFSET[1]:y,x-POS_OFFSET[0]:x+POS_OFFSET[0]]
                flat_search = search_area.reshape(POS_OFFSET[0]*POS_OFFSET[1]*2,3)
                
                contains_colour = (colour == flat_search).all(1).any()
                if contains_colour: 
                    blocks_with_colour.append((row_num, block_num))
                    if qbert_search:
                        return blocks_with_colour

                block_num += 1
            row_num += 1

        return blocks_with_colour
                
    def is_over(self):
        return self.ale.game_over()
    
    def reset(self):
        return self.ale.reset_game()
