import numpy as np

def q_func(game, weights=None, dist_func='euclid'):
    # Initializing weights and distances
    if weights == None:
        weights = np.random.uniform(-1,1,(6))

    goal_state_dist = get_goal_state_distance(game, dist_func)
    distances = get_distances(game, dist_func)
    distances = (1,goal_state_dist) + distances
    
    q_val = np.dot(weights, np.array(distances))
    return distances, q_val

def get_distances(game, dist_func):
    if dist_func == 'euclid':
        enemy_states_dist = get_euclid_dist(game, game.enemy_states)
        entity_states_dist = get_euclid_dist(game, game.entity_states)
        ldisc_dist = get_euclid_dist(game, [game.disc_states[0]])
        rdisc_dist = get_euclid_dist(game, [game.disc_states[1]])
    elif dist_func == 'manhattan':
        enemy_states_dist = get_manhattan_dist(game, game.enemy_states)
        entity_states_dist = get_manhattan_dist(game, game.entity_states)
        ldisc_dist = get_manhattan_dist(game, [game.disc_states[0]])
        rdisc_dist = get_manhattan_dist(game, [game.disc_states[1]])
    elif dist_func == 'hamming':
        enemy_states_dist = get_hamming_dist(game, game.enemy_states)
        entity_states_dist = get_hamming_dist(game, game.entity_states)
        ldisc_dist = get_hamming_dist(game, [game.disc_states[0]])
        rdisc_dist = get_hamming_dist(game, [game.disc_states[1]])
    return enemy_states_dist, entity_states_dist, ldisc_dist, rdisc_dist

def get_euclid_dist(game, states):
    dist = 0 
    # Get coords if not disc states
    if len(states) > 2:
        coords = game.get_coords_from_state(states)
    else:
        coords = states

    player_pos = np.array(game.player.pos)
    if len(player_pos) == 0: 
        return dist
    for coord in coords:
        if coord is not None:
            dist += np.linalg.norm(np.array(coord)-player_pos)
    return dist

def get_manhattan_dist(game, states):
    dist = 0 
    # Get coords if not disc states
    if len(states) > 2:
        coords = game.get_coords_from_state(states)
    else:
        coords = states

    player_pos = np.array(game.player.pos)
    if len(player_pos) == 0: 
        return dist
    for coord in coords:
        if coord is not None:
            dist += np.sum(np.fabs(np.array(coord)-player_pos))
    return dist

def get_hamming_dist(game, states):
    dist = 0 
    # Get coords if not disc states
    if len(states) > 2:
        val = 1
        coords = [(index, row.index(val)) for index, row in enumerate(states) if val in row]
    else:
        # -1 input as player pos can't match discs in y-direction
        coords = [(4, -1) for state in states if state is not None]

    player_pos = [(index, row.index(game.player.pos)) for index, row in enumerate(states) if game.player.pos in row]
    if len(player_pos) == 0:
        return dist
    else:
        player_pos = player_pos[0]
    for coord in coords:
        if coord is not None:
            if coord[0] != player_pos[0]:
                dist += 1
            if coord[1] != player_pos[1]:
                dist += 1
    return dist

def get_goal_state_distance(game, dist_func):
    state = game.get_state_rep()
    dist = 0

    if dist_func == 'euclid':
        for bit in state:
            dist += (1 - int(bit)) ** 2 
        dist = np.sqrt(dist)
    else:
        for bit in state:
            if bit == "0":
                dist += 1
    return dist

# TODO: Write function to execute an action and receive a reward
# TODO: Write get max Q-value function out of all actions
# TODO: Delta update rule
# TODO: Adding exploration 