import numpy as np
rng = np.random

rom_path = 'qbert.bin'
display_screen = True

try:
    from ale_py import ALEInterface
except ImportError as e:
    raise ImportError('Unable to import the python package of Arcade Learning Environment. ' \
                       'ALE may not have been installed correctly. Refer to ' \
                       '`https://github.com/mgbellemare/Arcade-Learning-Environment` for some' \
                       'installation guidance')

ale = ALEInterface()
ale.setInt(b'random_seed', rng.randint(1000))
if display_screen:
    import pygame
    pygame.init()

    ale.setBool(b'sound', False) # Sound doesn't work on OSX
    ale.setBool(b'display_screen', True)
else:
    ale.setBool(b'display_screen', False)
ale.setFloat(b'repeat_action_probability', 0)
ale.loadROM(str.encode(rom_path))

