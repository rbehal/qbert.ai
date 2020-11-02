from ale_py import ALEInterface
import numpy as np
import pygame

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
    
    def __init__(self,display=False,random_seed=123,frame_skip=5,rom_file='qbert.bin'):
        self.ale = ALEInterface()
        
        self.ale.setInt("random_seed", random_seed)
        self.ale.setInt('frame_skip', frame_skip)

        if display:
            pygame.init()
            self.ale.setBool('sound', sound)
            self.ale.setBool('display_screen', display)
        
        # Load the ROM file
        self.ale.loadROM(rom_file)
        
        # Set height and width, initialize RGB display array
        (h,w) = self.ale.getScreenDims()
        self.screen = np.zeros([h, w, 3], dtype=np.uint8)
        
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
        
    def update(self):
        self.ale.getScreenRGB(self.screen)
        self.update_goal_col()
        self.update_block_states()

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
#                 print(colour)
                if np.all(colour == self.goal_col):
                    self.block_states[row_num][block_num] = 1
                elif np.all(colour == self.goal_col):
                    self.block_states[row_num][block_num] = 0
                block_num += 1
            row_num += 1
                
    def is_over(self):
        return self.ale.game_over()
    
    def reset(self):
        return self.ale.reset_game()