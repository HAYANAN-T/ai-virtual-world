# When the AI Wins the Wrong Game

A small, reproducible Q-learning experiment accompanying the Medium article **The AI Was Winning. Nobody Had Told It How to Play Fair.**

## What this demonstrates

A tabular Q-learning agent moves along seven tiles (0–6). Tile 3 gives a bonus; tile 6 is the intended finish.

- **Proxy reward:** +1 whenever the agent re-enters tile 3, +5 for reaching tile 6, with no step penalty. The agent learns to repeat the bonus instead of finishing.
- **Corrected reward:** +1 only for the first bonus entry, +15 at the finish, and a -0.05 step cost. The agent learns to reach tile 6.

The two scenarios use the same seven-tile environment, 8,000 training episodes per seeded run, learning rate 0.16, discount factor 0.95, and 30 evaluation steps.

## Run locally

Requires **Python 3.9+** and **no external packages**.

```bash
cd reward_hacking_experiment
python reward_lab.py
```

The script prints the summary and regenerates `results.csv` and `summary.json` in this folder.

## Recorded results

| Scenario | Seeded training runs | Reached finish | Average bonus hits | Average steps |
| --- | ---: | ---: | ---: | ---: |
| Proxy reward | 20 | 0/20 | 14 | 30 |
| Corrected reward | 20 | 20/20 | 1 | 6 |

The results are a **synthetic reinforcement-learning demonstration**, not results from OpenAI's CoastRunners environment or a modern LLM. The corrected-reward scenario changes the reward design, not the task layout.

## Files

- `reward_lab.py` — complete experiment; standard-library-only implementation
- `results.csv` — all 40 recorded test runs
- `summary.json` — recorded aggregate results and representative trajectories

The original `reward_hack.py` in the repository root is preserved. No images, GIFs, article files, or publisher-sourced material are included here.
