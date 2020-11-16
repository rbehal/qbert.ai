from Game import Game
import numpy as np
import csv
import matplotlib.pyplot as plt

csv_name = "random"

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

num_episodes = 1500
game = Game()
actions = game.ale.getMinimalActionSet()
training_data = []

for episode in range(num_episodes):
    total_reward = 0

    game.initialize()
    while not game.is_over():
        action = np.random.choice(actions)
        reward = game.ale.act(action)
        
        game.update_RAM()

        reward += game.get_reward()
        total_reward += reward
        game.update()

    final_values = [episode, total_reward]
    training_data.append(final_values)
    print("Episode %d ended with score: %d" % (episode, total_reward))

    game.reset()

write_csv(training_data)
plot_training(training_data)