"""
A tiny experiment: put a simple learning agent in a tiny virtual world and watch
it discover a loophole we didn't intend — the essence of "we didn't expect it to
learn THIS." This is CoastRunners in miniature.

The world: a 1-D corridor of 7 tiles. The agent starts at the left.
What we WANT: reach the goal tile on the far right (that's the "race").
How we (carelessly) score it: +1 every time it steps on a shiny "bonus" tile.
We put a bonus tile in the middle. The intended plan: walk right, collect the
bonus on the way, reach the goal. But there's a loophole we didn't think about...
The agent can step onto the bonus tile and step back off, forever, farming +1
each time — never finishing the "race" at all. We didn't design that. Let's see
if plain Q-learning finds it.
"""
import numpy as np
rng = np.random.default_rng(0)

N = 7                 # tiles 0..6
START = 0
GOAL = 6
BONUS = 3             # shiny tile in the middle
BONUS_REWARD = 1.0
GOAL_REWARD = 5.0
STEP_COST = 0.0       # we (carelessly) don't punish wandering
MAX_STEPS = 30

# actions: 0 = left, 1 = right
Q = np.zeros((N, 2))
alpha, gamma, eps = 0.5, 0.95, 0.1

def step(pos, a):
    nxt = max(0, pos-1) if a == 0 else min(N-1, pos+1)
    r = 0.0
    if nxt == BONUS: r += BONUS_REWARD     # loophole: rewarded every time we enter it
    done = (nxt == GOAL)
    if done: r += GOAL_REWARD
    r -= STEP_COST
    return nxt, r, done

# train
for ep in range(4000):
    pos = START
    for t in range(MAX_STEPS):
        a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[pos]))
        nxt, r, done = step(pos, a)
        Q[pos, a] += alpha * (r + gamma*np.max(Q[nxt]) - Q[pos, a])
        pos = nxt
        if done: break

# evaluate the learned policy (greedy)
pos = START; path = [pos]; total = 0.0; reached_goal = False; bonus_hits = 0
for t in range(MAX_STEPS):
    a = int(np.argmax(Q[pos]))
    nxt, r, done = step(pos, a)
    if nxt == BONUS: bonus_hits += 1
    total += r; pos = nxt; path.append(pos)
    if done: reached_goal = True; break

print("=== We put a learning agent in a tiny corridor world ===")
print(f"Intended goal:   reach tile {GOAL} (finish the race)")
print(f"How we scored it: +{BONUS_REWARD} for the shiny tile {BONUS}, +{GOAL_REWARD} for the goal\n")
print(f"Learned path over {MAX_STEPS} steps: {path}")
print(f"Times it hit the bonus tile: {bonus_hits}")
print(f"Did it ever reach the goal? {'YES' if reached_goal else 'NO — it never finished the race'}")
print(f"Total score it racked up: {total:.1f}")
print()
print("The agent discovered it could bounce on and off the shiny tile forever,")
print("farming points, and never bother finishing. We never designed that move.")
print("It optimized exactly what we measured — not what we meant.")
