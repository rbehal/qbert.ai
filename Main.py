from Game import *
from QLearning import *
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

args = vars(ap.parse_args())

def main():
    print("""Running game with:
                        Seed: {}
                        Distance Function: {}
                        Exploration Function: {}
                        # of Episodes: {}
                """.format(args["seed"],args["distance"],args["exploration"],args["num_episodes"]))

    game = Game(display=args['display'], random_seed=args['seed'])
    learner = QLearning(game,dist_func=args['distance'],exp_func=args['exploration'])

    training_data = []
    minimal_actions = game.ale.getMinimalActionSet()

    for episode in range(args['num_episodes']):
        total_reward = 0
        count = 0 
        while not game.is_over():
            total_reward += learner.main()
            count += 1
            
        print(learner.weights)
        print("Episode %d ended with score: %d" % (episode, total_reward))
        
        output_list = [episode, total_reward] + list(learner.weights)
        training_data.append(output_list)
        
        game.reset()
        
    write_csv(training_data)

def write_csv(training_data):
    with open('training_data2.csv','w') as csv_file:
        headings = ["Episode Number", "Final Score"]
        weights =  ["constant", "goal state dist", "enemy state dist", "entity state dist", "ldisc dist", "rdsic dist"]
        headings = headings + weights

        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(headings)
        
        for data in training_data:
            writer.writerow(data)

if __name__ == "__main__":
    main()