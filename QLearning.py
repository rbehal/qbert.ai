from Game import Game
import numpy as np

class QLearning:
    def __init__(self,game,weights=None,dist_func='euclid',exp_func='eps-greedy',eps=0.05,temp=25,alpha=0.1,discount=0.995):
        self.game = game
        self.weights = weights
        self.dist_func = dist_func
        self.alpha = alpha
        self.discount = discount
        self.exp_func = exp_func
        self.eps = eps
        self.temp = temp

    def q_func(self, game):
        # Initializing self.weights and distances
        if self.weights is None:
            self.weights = np.random.uniform(-1,1,(6))

        distances = self.get_distances(game)
        
        q_val = np.dot(self.weights, np.array(distances))
        return distances, q_val

    def get_distances(self, game):
        if self.dist_func == 'euclid':
            enemy_states_dist = self.get_euclid_dist(game, game.enemy_states) / 150
            entity_states_dist = self.get_euclid_dist(game, game.entity_states) / 150
            ldisc_dist = self.get_euclid_dist(game, [game.disc_states[0]]) / 130
            rdisc_dist = self.get_euclid_dist(game, [game.disc_states[1]]) / 130
            goal_state_dist = self.get_nearest_targets_dist(game) / 2000
        elif self.dist_func == 'manhattan':
            enemy_states_dist = self.get_manhattan_dist(game, game.enemy_states) / 200
            entity_states_dist = self.get_manhattan_dist(game, game.entity_states) / 200
            ldisc_dist = self.get_manhattan_dist(game, [game.disc_states[0]]) / 166
            rdisc_dist = self.get_manhattan_dist(game, [game.disc_states[1]]) / 166
            goal_state_dist = self.get_nearest_targets_dist(game) / 2500
        elif self.dist_func == 'hamming':
            enemy_states_dist = self.get_hamming_dist(game, game.enemy_states) / 2
            entity_states_dist = self.get_hamming_dist(game, game.entity_states) / 2
            ldisc_dist = self.get_hamming_dist(game, [game.disc_states[0]]) / 2
            rdisc_dist = self.get_hamming_dist(game, [game.disc_states[1]]) / 2
            goal_state_dist = self.get_nearest_targets_dist(game) / 21
        
        # Add 1 for constant theta_0
        return 1, goal_state_dist, enemy_states_dist, entity_states_dist, ldisc_dist, rdisc_dist

    def get_euclid_dist(self, game, states):
        dist = 0 
        # Get coords if not disc states
        if len(states) > 2:
            coords = game.get_coords_from_state(states)
        else:
            coords = states

        player_pos = np.array(game.player.pos)
        if len(player_pos) == 0: 
            return dist
        # Calculate Euclidean distance 
        for coord in coords:
            if coord is not None:
                dist += np.linalg.norm(np.array(coord)-player_pos)
        return dist

    def get_manhattan_dist(self, game, states):
        dist = 0 
        # Get coords if not disc states
        if len(states) > 2:
            coords = game.get_coords_from_state(states)
        else:
            coords = states

        player_pos = np.array(game.player.pos)
        if len(player_pos) == 0: 
            return dist
        # Calculate manhattan distance
        for coord in coords:
            if coord is not None:
                dist += np.sum(np.fabs(np.array(coord)-player_pos))
        return dist

    def get_hamming_dist(self, game, states):
        dist = 0 
        # Get coords if not disc states
        if len(states) > 2:
            val = 1
            coords = [(index, row.index(val)) for index, row in enumerate(states) if val in row]
        else:
            # -1 input as player pos can't match discs in y-direction
            coords = [(4, -1) for state in states if state is not None]

        # Get player position in terms of indices --> [(y,x)]
        player_pos = [(index, row.index(game.player.pos)) for index, row in enumerate(game.BLOCK_POS) if game.player.pos in row]
        if len(player_pos) == 0:
            return dist
        else:
            player_pos = player_pos[0]
        # Calculate hamming distance
        for coord in coords:
            if coord is not None:
                if coord[0] != player_pos[0]:
                    dist += 1
                if coord[1] != player_pos[1]:
                    dist += 1
        return dist

    def get_nearest_targets_dist(self, game):
        dist = 0
        
        row_num = 0
        for row in game.block_states:
            block_num = 0            
            for block_state in row:
                if block_state == 0:
                    q_pos = np.array(game.player.pos)
                    block_pos = np.array(self.game.BLOCK_POS[row_num][block_num])

                    if self.dist_func == 'euclid':
                        dist += np.linalg.norm(q_pos - block_pos)
                    elif self.dist_func == 'manhattan':
                        dist += np.sum(np.fabs(block_pos - q_pos))
                    elif self.dist_func == 'hamming':
                        dist += 1

                block_num += 1
            row_num += 1

        return dist

    def get_max_q_action(self):
        minimal_actions = self.game.ale.getMinimalActionSet()
        minimal_actions.pop(1)
        
        action_values = {}
        action_str_to_ale_obj = {}
        for action in minimal_actions:
            act = str(action).split(".")[1] 
            
            temp_state = Game(gamestate = self.game)
            temp_state.execute_action(act)

            q_val = self.q_func(temp_state)[1]
            action_values[act] = q_val
            action_str_to_ale_obj[act] = action

        q_vals=list(action_values.values())
        acts=list(action_values.keys())
        best_action = acts[q_vals.index(max(q_vals))]
        # Return (Best Action ALE Object, Best Acction Q Value)
        return action_str_to_ale_obj[best_action], action_values[best_action]

    def get_eps_greedy_action(self):
        minimal_actions = self.game.ale.getMinimalActionSet()
        minimal_actions.pop(1)
        
        best_action = np.random.choice(minimal_actions)
        act = str(best_action).split(".")[1] 

        temp_state = Game(gamestate = self.game)
        temp_state.execute_action(act)
        
        q_val = self.q_func(temp_state)[1]
        return best_action, q_val

    def get_softmax_action(self):
        minimal_actions = self.game.ale.getMinimalActionSet()
        minimal_actions.pop(1)
        
        action_values = {}
        action_str_to_ale_obj = {}
        for action in minimal_actions:
            act = str(action).split(".")[1] 
            
            temp_state = Game(gamestate = self.game)
            temp_state.execute_action(act)

            q_val = self.q_func(temp_state)[1]
            action_values[act] = q_val
            action_str_to_ale_obj[act] = action

        values = np.array(list(action_values.values())) / self.temp
        probabilities = np.exp(values)
        probabilities = probabilities / probabilities.sum()

        best_action = np.random.choice(list(action_values.keys()), p=probabilities)
        # Return (Best Action ALE Object, Best Acction Q Value)
        return action_str_to_ale_obj[best_action], action_values[best_action]

    def update_weights(self, curr_state_q, curr_state_fevals, best_action, reward):
        self.game.update()
        
        pre_factor = self.alpha * (reward + self.discount*best_action[1] - curr_state_q)

        self.weights = np.add(self.weights, pre_factor * curr_state_fevals)
        return 
        
