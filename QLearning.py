def q_func(state, weights=None, dist_func='euclid'):
    # Initializing weights
    if weights == None:
        weights = np.random.uniform(-1,1,(2))
    
    if dist_func == 'euclid'
        dist = get_euclid_dist(state)
    elif dist_func == 'hamming'
        dist = get_hamming_dist(state)
    else
        dist = get_jaccard_dist(state)

    q_val = weights[0] + weights[1] * dist
    return q_val

def get_euclid_dist(state):
    dist = 0
    for bit in state:
        dist += (1 - int(bit)) ** 2 
    dist = np.sqrt(dist)
    return dist

def get_hamming_dist(state):
    dist = 0
    for bit in state:
        if bit == "0":
            dist += 1
    return dist

def get_jaccard_dist(state):
    intersection = 0
    for bit in state:
        if bit == "1":
            intersection += 1
    dist = 1 - intersection / len(state)
    return dist

# TODO: Write function to execute an action and receive a reward
# TODO: Write get max Q-value function out of all actions
# TODO: Delta update rule
# TODO: Adding exploration 