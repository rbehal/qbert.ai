class Player:
    def __init__(self, ale, player=None):
        if player is not None:
            self.lives = player.lives
            self.pos = player.pos
            return
        self.ale = ale
        
        self.lives = ale.lives()
        self.pos = (0, 0)