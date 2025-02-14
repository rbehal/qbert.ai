# Usage:
# 
# python3 Main.py -d <displayOn?> -s <seed> -m <dist_func> -x <exp_fuc> -t <approx_type> -n <num_episodes> -f <csv_name> -w <weights>


# Imports 
from Game import *
from QLearning import *
import matplotlib.pyplot as plt
import argparse
import logging
import csv

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", default=False, type=bool)
ap.add_argument("-s", "--seed", default=123, type=int)
ap.add_argument("-m", "--distance", default='euclid', choices=['euclid','manhattan','hamming'])
ap.add_argument("-x", "--exploration", default='eps-greedy', choices=['eps-greedy','softmax'])
ap.add_argument("-t", "--approx_type", default='complex', choices=['simple','complex','mixed'])
ap.add_argument("-n", "--num_episodes", default=500, type=int)
ap.add_argument("-f", "--csv_name", default='training_data')
# Should be input as the file path of a JSON file containing an array
ap.add_argument("-w", "--weights")

# Set argument variables
args = vars(ap.parse_args())
display = args['display']
seed = args['seed']
exp_func = args['exploration']
dist_func = args['distance']
approx_type = args['approx_type']
num_episodes = args['num_episodes']
csv_name = args['csv_name']
weights = args.get('weights')



def main():
    # Initialize game and learner
    game = Game(display=display, random_seed=seed)
    learner = QLearning(game, dist_func=dist_func, exp_func=exp_func, approx_type=approx_type, weights=weights)

    # Display run parameters
    print("""Running game with:
                        Seed: {}
                        Distance Function: {}
                        Exploration Function: {}
                        Approximation Function: {}
                        # of Episodes: {}
                        CSV Filename: {}
                """.format(seed,dist_func,exp_func,approx_type,num_episodes,csv_name))

    # Initialize data structure for CSV, list of minimal actions, and stats
    training_data = []
    stats = [] 
    minimal_actions = game.ale.getMinimalActionSet()
    minimal_actions.pop(1)

    # Start training
    for episode in range(num_episodes):
        # Initialize reward
        total_reward = 0
        count = 0 

        game.initialize()
        while not game.is_over():
            # Get current state q_values and grad_theta_q values
            curr_state_q = learner.q_func(game)[1]
            curr_state_fevals = np.array(learner.get_distances(game))
            
            # Get action based on exploration strategy
            if learner.exp_func == "eps-greedy" and np.random.random() < learner.eps:
                best_action = learner.get_eps_greedy_action()
            elif learner.exp_func == "softmax":
                best_action = learner.get_softmax_action()
            else:
                best_action = learner.get_max_q_action()

            # Execute action and update weights based on reward
            reward = game.ale.act(best_action[0])
            game.update_RAM()
            
            reward += game.get_reward(reward)
            total_reward += reward
            
            learner.update_weights(curr_state_q, curr_state_fevals, best_action, reward)
            count += 1
        
        # End of episode print information
        print(learner.weights)
        print("Episode %d ended with score: %d" % (episode, total_reward))
        
        # Append data to array for CSV writing
        final_values = [episode, total_reward] + list(learner.weights)
        training_data.append(final_values)
        
        game.reset(total_reward)
    
    # Display overall game analysis
    scores = game.high_scores
    print("""Game analysis:
                        Highest score: {}
                        Highest # of Sams killed: {}
                        Highest # of Coilys killed: {}
                        Highest # of green balls caught: {}
                """.format(scores[0],scores[1],scores[2],scores[3]))
    write_csv(training_data)
    plot_training(training_data)

# Writes csv file with training data (game scores and weights)
def write_csv(training_data):
    with open(csv_name + ".csv",'w') as csv_file:
        # Initialize headers for CSV file
        headings = ["Episode Number", "Final Score"]
        weights =  ["constant", "goal state dist", "enemy state dist", "entity state dist", "ldisc dist", "rdsic dist"]
        headings = headings + weights
        # Writing headers
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(headings)
        # Writing data
        for data in training_data:
            writer.writerow(data)

# Displays graph of episode number vs. score
def plot_training(training_data):
    x = []
    y = []
    for episode in training_data:
        x.append(episode[0])
        y.append(episode[1])
    plt.plot(x, y)
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.show()
    plt.savefig(csv_name)

if __name__ == "__main__":
    main()