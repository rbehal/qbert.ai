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
ap.add_argument("-n", "--num_episodes", default=500, type=int)
ap.add_argument("-f", "--csv_name", default='training_data')

args = vars(ap.parse_args())
display = args['display']
seed = args['seed']
exp_func = args['exploration']
dist_func = args['distance']
num_episodes = args['num_episodes']
csv_name = args['csv_name']

def main():
    # Initialize game and learner
    game = Game(display=display, random_seed=seed)
    learner = QLearning(game, dist_func=dist_func, exp_func=exp_func)

    # Display run parameters
    print("""Running game with:
                        Seed: {}
                        Distance Function: {}
                        Exploration Function: {}
                        # of Episodes: {}
                        CSV Filename: {}
                """.format(seed, dist_func, exp_func, num_episodes,csv_name))

    # Initialize data structure for CSV and list of minimal actions
    training_data = []
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
            reward += game.get_reward()
            
            learner.update_weights(curr_state_q, curr_state_fevals, best_action, reward)

            total_reward += reward
            count += 1
        
        if episode % np.floor((num_episodes / 10)) == 0:
            if learner.alpha > 0.0001:
                learner.alpha = learner.alpha / 2

        print(learner.weights)
        print("Episode %d ended with score: %d" % (episode, total_reward))
        
        # Append data to array for CSV writing
        final_values = [episode, total_reward] + list(learner.weights)
        training_data.append(final_values)
        
        game.reset()
    write_csv(training_data)
    plot_training(training_data)

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

def plot_training(training_data, num_past_avg = 10):
    x = []
    y = []
    
    avg = []
    count = 0
    for episode in training_data:
        count += 1
        ep_num = episode[0]
        score = episode[1]
        
        avg.append(score)
        if count == num_past_avg:
            count = 0
            
            x.append(ep_num)
            y.append(sum(avg)/len(avg))
            
            avg = []
 
    plt.plot(x, y)
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.show()
    plt.savefig(csv_name)

if __name__ == "__main__":
    main()