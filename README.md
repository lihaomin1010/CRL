# Pytorch Implementation Contrastive Reinforcement Learning
This repository includes a pytorch implementation of Contrastive Learning as Goal-Conditioned RL. 

## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)
- [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay)
- [Contrastive Learning for Goal-Conditioned RL(https://arxiv.org/abs/1707.01495)](https://github.com/google-research/google-research/tree/master/contrastive_rl)

## Instructions
1. Create a Conda Environment: ``` conda create -n equ_contrastive_rl_env python=3.9 ```
2. Activate the Conda environment: ``` conda activate equ_contrastive_rl_env ```
3. Install Pip < 24.1: ``` pip install pip==23 ```
4. Install Setuptools == 66: ``` pip install setuptools==66 ```
5. Install OpenAI Gym == 0.19.0: ``` pip install gym==0.19.0 ```
6. Install Mujoco210 as per the instructions in https://github.com/openai/mujoco-py
7. Install the remaining dependencies: ``` pip install -r requirements.txt --no-deps ```
8. Install mpi4py: ``` conda install mpi4py ```
9. Install Pytorch




## Instruction to run the code
If you want to use GPU, just add the flag `--cuda`. If you want to use the equivariant version, add the flag `--equivariant`. Use the following examples to run your experiments:
1. Train the contrastive-rl agent on the **State-Based FetchPush** task::
```bash
python train_contrastive_state_based.py --env-name=fetch_push  --n-epochs=200 --cuda --seed=0
```
2. Train the contrastive-rl agent on the **Image-Based FetchReach**:
```bash
python train_contrastive_img_based.py --env-name=fetch_reach  --n-epochs=20 --cuda --seed=0
```
