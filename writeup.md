# How I made Ender

Hi everyone! This has been my first Kaggle competition, and it's been a blast. I'd like to say a big thank you to the hosts, and to all the other competitors who have made this competition so fun, challenging and rewarding. I'm hoping to participate in many more of these simulation competitions in the future.

I am proud to say that I completed this project from a position of relative 'compute poverty'. I have been experimenting and training on my own ageing 3080 10GB and 11 days of a rented 4090 instance, totaling around $170 plus electricity. My final 2-player training run took 3.4 days on the 4090 to the checkpoint I used in my submissions, costing about $51 (discounting an experimental fork I ended up discarding). My 4-player model was trained entirely on my 3080.

In this document, I'll do my best to outline all the important aspects of my approach that contributed to Ender's strength.

## Overview

Ender is an RL solution, trained purely using self-play and leagues made of my own past checkpoints. I never used the daily episode datasets. I never trained against any public agents. I used separate models for 2-player and 4-player game modes.

Ender's backbone is a transformer, with one CLS token and a token for each planet. It is based on the observation that making the next decision after committing to a launch is strategically the same as if the new fleet were already in play at the start of the turn. A turn goes as follows:

1. Decide globally whether to launch or halt.
2. If launch, sample an origin and send fraction from a joint distribution.
3. For the chosen origin-fraction, compute reachable targets & ETAs.
4. Sample from the candidate targets, with an additional 'abort' option.
5. Add the new fleet to the observation as if it were already in play. In the 'abort' case, mask the origin-fraction for the rest of the turn.
6. Repeat up to 16 times until halt.

I'll refer to each iteration within a game turn as a 'micro-step'.

Ender was trained using PPO and GAE. Each micro-step is treated as one step in the trajectory. The game turn is given as part of the observation but the RL is all done in terms of micro-steps.

Ender uses some test-time search/planning, which I have found to slightly improve ladder performance, particularly in 4-player games.

## Features

The following features are included in the input for each token:

1. Entity type (planet/comet/CLS)
2. Owner
3. Production
4. Garrison
5. Velocity
6. Radius
7. Turn index
8. Net incoming fleets
9. Fleet survivor owners (4p only)
10. Which fractions are blocked by a prior abort
11. Projected future garrisons and owners assuming ceasefire

The position of each planet is encoded using 2D RoPE.

The target MLP takes as input:

1. Hidden state of the target (transformer output)
2. Fleet size
3. ETA
4. Whether fleet size is greater than target garrison (boolean)
5. Projected future garrisons and owners of both origin and target assuming ceasefire after fleet launch

I normalise observations to the p0 perspective. Seats other than p0 are transformed to appear as if they were from the p0 perspective. This affects opponent labeling, positions and velocities.

### Incoming fleets

Planets do not care which origin an inbound fleet is coming from. They also do not care about multiple fleets arriving on the same turn. What's important is the net effect. I resolve inter-fleet combat according to the game rules before computing incoming fleet features.

Incoming fleets are encoded as 24 bins indicating the net effect of fleets arriving in the next 24 turns (positive for friendly, negative for enemy). In the 4-player version, there are also 24 one-hot bins for the surviving fleet owner to differentiate between enemies.

### Future features

Before adding the future features, I had a big problem: the policy would often launch an attack slightly too early (before it has enough ships) or slightly too late (losing the initiative). It wasn't confident on whether a fleet is exactly enough to capture a target. For example, if you start on a production 1 planet, and there's a juicy production 5 neutral planet nearby with a small garrison, your best strategy is to wait until you have 1 more ship than the target, then launch. If you wait any longer, your opponent can gain an immediate advantage by capturing sooner. If you launch too early, it's even worse: your next fleet will take ages to reach the target. I initially tried to paper over this with test-time search, which works, but tends to burn through a lot of overage time.

Adding future features to the target MLP fixed this: if the policy doesn't like the projected outcome, it can just abort.

Adding future features to planets provides additional advantages. It allows the policy to easily see if and when planets are due to be captured, and can allocate attacks and reinforcements more precisely and efficiently. It also allows the value head to respond immediately and confidently to good and bad launches in training.

Like incoming fleets, future features have 24 bins. These contain the projected owner and garrison for the next 24 turns.

## Action space

See the overview for an outline of the action space.

I decided to make my action space as expressive as possible. Since I am not an expert Orbit Wars player, I can't say with confidence that there's no benefit to e.g. launching fleets smaller than full send, launching multiple fleets from the same origin on the same turn, etc. So I designed the action space such that the model can choose any sequence of actions the game allows, except:

- Launches choose from 5 fractions of the garrison to send (0.2, 0.4, 0.6, 0.8, 1.0)
- Only up to 16 launches are allowed per turn

Even with these restrictions, I believe the action space is expressive enough that it never or very rarely becomes the bottleneck.

As it turns out, Ender does indeed use the ability to launch multiple fleets from the same origin, and to great effect.

### Why not just sample from the whole distribution of possible launches, with a noop?

Because with up to 44 planets and 5 send fractions, this quickly blows up into an unreasonable amount of computation. The target MLP needs to know the ETA and future projections, which means we'd have to compute the flight path for potentially 44 * 43 * 5 = 9,460 launch candidates every micro-step. My factorisation prunes this to just 43.

## Architecture

Ender's core is a basic transformer encoder with 4 layers and d_model 192. I have experimented with larger models but found they learn slower and don't show any meaningful benefit. More compute might have changed things, however.

The features are put through embeddings or linear transformations directly into the input tokens.

The value head, launch/halt head, origin-fraction head and abort head are all just linear layers from planet hidden states.

The target scorer (MLP(483, 192, 96, 1) for each candidate target) consumes both the target's hidden state and the additional features associated with that launch candidate. It does not consume the origin hidden state, because the target does not need to know where a candidate fleet is coming from to evaluate its benefit.

## Training

I trained using PPO on self-play and leagues of past checkpoints. I ported the game engine to JAX to get the necessary throughput. Hyperparameters were:

- Learning rate: initially 3e-5, periodically reduced
- Entropy coefficient: initially 0.01, periodically reduced
- Parallel envs: 384 for 4p, 2048 for 2p
- Rollout length: 512 micro-steps for 4p, 256 for 2p
- Gamma: 0.998
- Lambda: 0.95
- Clip epsilon: 0.2
- PPO epochs: 2
- Minibatch size: 2048
- Max grad norm: 0.5

I used a reward of 0 for a loss, 1 for a draw and 2 for a win. I avoided negative rewards because with gamma < 1, that would encourage 'delaying losing' rather than just trying to maximise winning. Such stalling hurts game throughput. I did also try setting gamma == 1, which resulted in a policy that does nothing once it has a big enough advantage (no decisive victories), like Isaiah. However it also did nothing if it was significantly behind, so most of the samples were just one seat comfortable in its lead and one that has given up, both doing nothing and wasting throughput. Also, at test time, it was less than perfect at maintaining a lead for a full game against an opponent who actually fights back.

The 2-player model was trained on about 3.1B samples over 1B env steps. The 4-player model was trained on about 1.5B samples over 413M env steps.

### Entropy

I initially had a lot of trouble with agents insisting on sending large numbers of small fleets. This hurt training for 2 reasons:

1. It's rarely a good strategy.
2. With so many launches, credit assignment is messier.

I tried initialising the model with a higher halt likelihood and a full-send bias, but it quickly regressed to the same behaviour. I eventually had the insight that a standard entropy bonus might be the wrong tool for the job. Who says the 'default' should be all options being equally likely? For launch/halt and for the fraction dimension of origin-fraction, I replaced the entropy bonus with a KL penalty for divergence from the initial prior. I use halt_init_prob=0.9, fraction_init_ratio=1:1:1:1:10. This resulted in a huge immediate improvement.

So the 'entropy coefficient' is something of a misnomer in this case, as it's applied to the KL penalty too.

I found that reducing this coefficient later in training resulted in significant improvement. Even setting it to 0 worked well in some cases, but didn't seem to generalise quite as well to the ladder.

### Earlygame envs

When training in the 4-player game mode, games would often run for the full 500 turns. This caused significant mid- to late-game bias in training, and the critical earlygame was neglected. In 2-player, this wasn't so much of a problem, as games ended much faster.

To address this, I set aside half the environments in the 4-player run to be truncated (with value bootstrap) after 50 turns, so there is always plenty of early-game data.

### League

To try and prevent overfitting to playing against the current policy, some envs were dedicated to playing against past checkpoint of self, plus a couple of strong checkpoints from prior experiments. These were prioritised based on recent winrate, with more difficult checkpoints being played more often. This seems especially important for the 4-player game mode.

Aside from improving the policy's ability to generalise, this also makes it easy to see whether it's still improving. Each checkpoint's ELO can be estimated from its wins and losses against other checkpoints.

### Environment

A key optimisation I made was to notice that fleet paths are already calculated at launch time, so the JAX port of the environment doesn't actually need to simulate fleet movement or track fleet position. It can just have incoming bins for each planet, and shift left every turn.

Porting the thing to JAX was a headache, but it paid dividends: I got to around 15k micro-steps per second on the 4090, peaking at roughly 4k env steps per second, including PPO. Less than some others have managed, but on weaker hardware and with my micro-steps approach it was about as good as I could manage.

## Competition Harness

My two final submissions use the same models but two different configurations:

1. Sampled action search for 4p, plain policy (sampled) for 2p
2. Launch/halt search for both game modes

I'm not entirely convinced that there's a benefit to search in 2p, so I split my strategy.

### Sampled action search

This just samples the full turn of actions 10 times, samples a few sets of opponent actions using a small distilled policy, does one simulated env step and picks whichever action sequence has the highest mean value over the different opponent actions.

### Launch/halt search

This is a deeper search that only runs on the first non-abort micro-step of each turn. It has 2 long branches: halt and launch. Each branch follows the policy greedily until the launch fleet hits and for 3 turns after. The distilled policy is used for opponent actions and some simulated future ego actions to save time.

### Launch geometry

In training, I simply simulate fleets being launched in 256 directions in parallel, and see what they hit. On a GPU, this is fast. On a slow CPU, it's not quite good enough if you want to search.

In the competition harness, I instead model the fleet as an expanding circle, and solve for tangencies with planets to detect collision windows. There is a closed-form solution for both stationary and moving planets if you model the motion as a polyline (which aligns with the collision maths in the official env). I then solve for angular extrema within each collision window using sextic roots (numerical method). To be quite honest, I don't fully understand the maths, but GPT does and it works well. I then do an occlusion sweep so I know for any angle where it will hit and approximately when. I finally simulate the launch to get an exact ETA.

## GG everyone!
