# Platformer Game

Platformer Game is a game that explores machine learning and how to train agents more effectively. As you get better at the game, you get better at training ML agents!

You have to train an ML agent to play through a large platformer level. It would take very long for the agent to learn how to complete the whole level by itself because there is no way of knowing if what it is doing is good unless it completes the whole level by chance. This is very unlikely, so we need to help it.

We can supply shorter levels that it is more likely to complete through random chance, which gives it a starting point from which it can learn.
You can also add diamonds to reward the agent for doing things that you want it to, this points it in the right direction to complete the whole level.

Once your agent has trained for about 3 minutes, we'll then test the agent to see if it can complete the whole level. We'll try it 1000 times - If you get a 95% completion rate then you have passed the level.

That's a hard challenge in itself, but that's not all.
We'll give feedback on how efficient your training set was. In particular, we'll show how many levels you used to train your agent and how many diamonds you used over all of your levels. In both cases, the fewer the better.

## Setup

First make sure Python 3.6.2 is installed. You cannot currently use 3.7. On Windows, you can get 3.6.2 from this [direct link](https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe) or if you want a different installer you can choose [here](https://www.python.org/downloads/release/python-362/). When installing it, make sure you select the "Add to environment variables" checkbox.

Run the following command on your command line to install Python dependencies
```
python -m pip install --upgrade pip
pip install gym matplotlib h5py "numpy>=1.14.0,<=1.14.5" Pillow chainer>=4.1.0 chainerrl>=0.3.0 tensorflow>=1.5.0 pygame
```

Download [Tiled level editor](https://www.mapeditor.org/).

Try dragging and dropping the "PlatformerGame/Demo/complete" folder onto the "PlatformerGame/attempt_goal_1.bat" script. If it starts up and shows a gameboy guy running around then you're all set!

## Gameplay

In Tiled, open "PlatformerGame/Data/Test/goal1.json". Your agent needs to complete this level.

Use the level editor to create your training levels. These should be 7x5 tiles large. We don't check this, so it's up to you not to cheat! Use 64x64 tile size when creating your level, and use the "Data/PlatformerGame.tsx" tileset. You will be prompted to save these in the .tmx format, but you will need to use the "File" menu to export them to .json files. Create a new folder and save all your .json files into it.

You can create any levels you want. There should only be one player start location per world, but you can have as many exits as you like. The two different ground tiles are interchangeable (whichever type you use, we'll try to make it look pretty in the game).

The agent can see the ground tiles and the exit, but cannot see the diamonds - if you put a diamond over a gap, the agent will collect the diamond when they jump there. Because they can't see the diamond, they will associate the reward with jumping over the gap and do that more often.

Once you have saved your levels as .json files in their own folder you can use the trigger_training.bat script to start the game. Drag your folder onto trigger_training.bat. This will start the training, and you will be shown the agent learning over your training levels! The agent will do 100 training "epochs" (each epoch is a run through all your levels).

Once the agent has finished training, it will attempt the goal level 1000 times, and you will be presented with how often your agent wins the level. You'll see your agent attempting the level, so you can spot areas where it doesn't know what to do and create new training levels to fix it. You'll then see a end-of-game score sheet which will show your win rate, how many levels used and how many diamonds used. Edit your levels and try again!

## Extensions and Future Work

This game is a prototype and could be improved in its game loop polish and in the amount of content it offers. As it stands, it does a good job of proving that the game is fun.

### Polish

The game loop feels very prototype-y because you have to edit in one program and see progress in another. The editor quality in Tiled is great, so we'd need to work hard to get to a similar level of quality. Using our own editor has other advantages such as forcing the use of our tileset and the right sized tiles, which might cause confusion if they were accidentally changed.

Adding a highscore server would add to the self-improvement motivator. This is intended to be presented similarly to the Zachtronics score histograms, where you yourself could decide at which point you are happy with how good your solution is. This would require fixed initialisation weights for the neural nets.

There are many other game-focused general improvements, such as adding sound effects and music, or adding animations to the UI.

### Concept Extension

The game as it stands teaches the player about a single fully connected neural network. This is a great starting point, but there are many other concepts in machine learning than simply the training data.

#### More levels, Indirectly Teaching Overfitting

One very simple next step would be to teach the player about overfitting, without saying the word overfitting at all. Instead, we'd give the players a few levels to complete, similar in difficulty to the current level. They would all start the player on the left and place the goal on the right (the brain and the observations are not enough to learn to search in both directions).
Once the player has trained individual agents to complete each level, we would then ask the player to train a single agent that can complete all levels. The players will find this more difficult than the single levels, and therefore learn the **intuition** about overfitting. Once the players understand the concept, we can name it for them.

#### Campaigns

As a further extension, I would recommend adding new "campaigns". Each campaign would focus on one type of brain and how to work with it, the first campaign would consist of the levels as described above. The second campaign would use a more complicated brain that takes the same observations of the world as the first brain, but also receives another signal that indicates whether the agent needs to primarily move right, left, up, or down to reach the exit. The levels match these new inputs, so the exit might be above, below, or to the left or the right of the player's start location.

A third campaign would take the agent the player trained in campaign two, and we would feed it a larger area of the level. It would use convolutional neural nets to produce the directional input for the agent from campaign two.

Other campaigns could focus on different things, such as hyperparameter design, curriculum learning, imitation learning, genetic algorithm based brains, etc.

#### User Generated Content

Within a campaign, players could generate new levels for each other to complete. This has been proven to extend the longevity of Zachtronics games, so may encourage further play with this game.
