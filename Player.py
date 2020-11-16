class Player:
    def __init__(self, ale, player=None):
        if player is not None:
            self.lives = player.lives
            self.pos = player.pos
            self.alive = True
            return
        self.ale = ale
        self.alive = True
        self.lives = ale.lives()
        self.pos = (76, 35)
    def died(self):
        self.lives -= 1
        self.alive = False