# Active Reinforcement Learning for Atari 2600 Game Q\*bert

The following code serves to be an implementation of Q-Learning with function approximation to play Q\*bert in the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment). The rules and guidelines for the game can be found in this [Atari Magazine](https://www.atarimagazines.com/cva/v1n2/qbert.php).


## Playing the game
In order to have the client play the game, ensure the following packages are installed with Python 3:

- pygame
- numpy
- matplotlib.pyplot
- ALE (Arcade Learning Environment)

Include the following files in the same directory:

- Main.py
- Game.py
- Player.py
- QLearning.py
- qbert.bin

The agent and learning can be launched from the command-line through the following:
~~~
python3 Main.py <optional flags>
~~~

An explanation of flags, possible input parameters, and default values are listed as follows: 

| Parameter    | Flag | Default       | Options                                                                           |
|--------------|------|---------------|-----------------------------------------------------------------------------------|
| display      | -d   | False         | Boolean determining whether a display should pop up.                              |
| seed         | -s   | 123           | Seed of the game. Should be a positive integer.                                   |
| distance     | -m   | euclid        | Distance function used. Options: [euclid,manhattan,hamming]                       |
| exploration  | -x   | eps-greedy    | Exploration function used. Options: [eps-greedy,softmax]                          |
| approx_type  | -t   | complex       | Type of function approximation. Options: [simple,mixed,complex]                   |
| num_episodes | -n   | 500           | Number of game episodes to train through/play. Should be a positive integer.      |
| csv_name     | -f   | training_data | Name of CSV and image file output in the directory containing training data.      |
| weights      | -w   | None          | Path of .json file containing an array of weights. If None, randomly initialized. |

## Examples
~~~
python3 Main.py -x softmax -n 1500
~~~

~~~
python3 Main.py -t mixed -x softmax -n 2000 -s 9999 -f softmax_mixed
~~~

~~~
python3 Main.py -t simple -m manhattan -f manhattan_simple -w model.json -d True
~~~
