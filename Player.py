class Player:
    def __init__(self, ale):
        self.ale = ale
        
        self.lives = ale.lives()
        self.pos = (0, 0)