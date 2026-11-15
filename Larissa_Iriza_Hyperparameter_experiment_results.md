# Hyperparameter Experiments

Member Name: Larissa Iriza

Each experiment changes one value from the baseline (Experiment 1), so each difference can be linked to that value. My experiments focus on gamma and the epsilon settings. Every run trained for 500,000 steps on Kaggle with the same seed. The average reward is over the last 100 episodes.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Baseline. Average reward −17.5 (363 episodes). Flat near −20.5 until about 230k steps, then rose steadily and was still rising at the end. |
| 2 | 0.0001 | 0.999 | 32 | 1.0 | 0.01 | 100000 | Average reward −16.2 (358 episodes). Started improving earlier than the baseline (about 170k steps) and finished a little higher. Valuing far-future rewards more did not hurt and helped slightly. |
| 3 | 0.0001 | 0.98 | 32 | 1.0 | 0.01 | 100000 | Best run. Average reward −12.2 (333 episodes, so the longest games). Rose first (about 140k steps) and fastest, and was still rising. A slightly shorter horizon gives smaller, less noisy Q-value targets, which seems to help learning in Pong, where each point is decided quickly. |
| 4 | 0.0001 | 0.85 | 32 | 1.0 | 0.01 | 100000 | Average reward −17.5 (437 episodes). Started improving late (about 250k steps). More episodes means shorter games: with a very low gamma the agent hardly values rewards more than a few steps ahead, so it struggles to plan its returns. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Average reward −14.4 (421 episodes). Flat until about 200k steps, when exploration ended, then climbed the most steeply of all runs. The extra exploration filled the replay buffer with more varied experience, which paid off once the agent started exploiting. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 400000 | Average reward −18.0 (420 episodes). The slowest learner: epsilon stayed high until 400k steps, so most of training was spent on random moves and there was little time left to use what it learned. |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 25000 | Average reward −16.9 (333 episodes). Started improving early (about 160k steps) because it stopped exploring quickly, but the climb slowed near the end, likely because it explored too little to find better strategies. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 100000 | Average reward −17.7 (410 episodes). Similar to the baseline but levelled off near the end. With 20% random moves even after training, the agent keeps missing shots it could have returned. |
| 9 | 0.0001 | 0.99 | 32 | 0.7 | 0.01 | 100000 | Average reward −17.7 (361 episodes). Almost the same as the baseline. Starting with less exploration made little difference because epsilon reaches 0.01 at 100k steps either way. |
| 10 | 0.0002 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Worst run. Average reward −20.9 (624 episodes, so very short games). Never learned and stayed near −21 for the whole run. Doubling the learning rate made the updates too large and unstable for the network to learn. |

## Best and worst configuration

My best experiment was Experiment 3 (gamma 0.98), which reached an average reward of −12.2 and started learning earliest. My worst was Experiment 10 (learning rate 0.0002), which never learned at all. This shows that DQN is very sensitive to the learning rate. Among the epsilon settings, a moderately slower decay (Experiment 5, 200,000 steps) worked best, while decaying too slowly (Experiment 6) wasted training time on random moves.

The configuration I would choose is learning rate 0.0001, gamma 0.98, batch size 32, epsilon from 1.0 to 0.01, with an epsilon decay of 200,000 steps. This combines my two best changes, although I did not test them together. Each configuration ran once with one seed, and my laptop baseline (−14.3) differed from the Kaggle baseline (−17.5) by about 3 points, so small differences between runs may be due to chance. The large differences (Experiments 3, 5 and 10) are clear.
