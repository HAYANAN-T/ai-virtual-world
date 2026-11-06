"""Seven-tile proxy reward experiment. Run python reward_lab.py; standard library only."""
import csv, json, random, statistics
from pathlib import Path

N=7
BONUS=3
GOAL=6


def play(policy, shaped, rng, max_steps=30):
    pos=0; visited_bonus=False; total=0.; hits=0; trace=[pos]
    for step in range(max_steps):
        action=policy[pos] if rng.random()>0.0 else rng.choice([-1,1])
        new=max(0,min(GOAL,pos+action))
        entered=(new==BONUS and pos!=BONUS)
        if shaped:
            bonus=1.0 if entered and not visited_bonus else 0.
            rew=bonus - .05 + (15.0 if new==GOAL else 0.)
        else:
            rew=(1.0 if entered else 0.)+(5.0 if new==GOAL else 0.)
        if entered: hits+=1; visited_bonus=True
        total+=rew; pos=new; trace.append(pos)
        if pos==GOAL: break
    return {'finished': pos==GOAL, 'steps':len(trace)-1,'bonus_hits':hits,'return':round(total,3),'trace':trace}


def train(seed, shaped=False, episodes=8000, max_steps=30):
    rng=random.Random(seed)
    q=[[0.,0.] for _ in range(N)]
    alpha=.16; gamma=.95
    for ep in range(episodes):
        pos=0; visited_bonus=False
        eps=max(.035, .90 * (1-ep/episodes)**1.7)
        for step in range(max_steps):
            if rng.random()<eps:
                action=rng.randrange(2)
            else:
                mx=max(q[pos]); best=[i for i,v in enumerate(q[pos]) if abs(v-mx)<1e-10]; action=rng.choice(best)
            new=max(0,min(GOAL,pos+(-1 if action==0 else 1)))
            entered=(new==BONUS and pos!=BONUS)
            if shaped:
                bonus=1.0 if entered and not visited_bonus else 0.
                reward=bonus-.05+(15.0 if new==GOAL else 0.)
            else:
                reward=(1.0 if entered else 0.) + (5.0 if new==GOAL else 0.)
            visited_bonus |= entered
            target=reward if new==GOAL else reward+gamma*max(q[new])
            q[pos][action]+=alpha*(target-q[pos][action])
            pos=new
            if pos==GOAL: break
    policy=[-1 if vals[0]>vals[1] else 1 for vals in q]
    policy[GOAL]=0
    return policy,q


def main():
    records=[]
    for scenario in ('proxy','corrected'):
        for seed in range(20):
            policy,q=train(100+seed,shaped=(scenario=='corrected'))
            outcome=play(policy, scenario=='corrected', random.Random(seed))
            records.append({'scenario':scenario,'seed':seed,**outcome,'policy':policy})
    path=Path(__file__).parent
    with (path/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenario','seed','finished','steps','bonus_hits','return','trace','policy']);w.writeheader();w.writerows(records)
    summary={}
    for scenario in ('proxy','corrected'):
        r=[x for x in records if x['scenario']==scenario]
        summary[scenario]={'seeds':len(r),'finish_rate':sum(x['finished'] for x in r)/len(r),
                           'mean_bonus_hits':round(statistics.mean(x['bonus_hits'] for x in r),2),
                           'mean_steps':round(statistics.mean(x['steps'] for x in r),2),
                           'representative_seed0':r[0]}
    (path/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
