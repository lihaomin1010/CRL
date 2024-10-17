import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc, rcParams
import matplotlib

matplotlib.use("Agg")


def generate_equi_contrastive_curves_arrays(runs):
    all_rewards = []
    all_steps = []

    for x, run in enumerate(runs):
        configs = {k: v for k, v in run.config.items() if not k.startswith("_")}
        history = run.scan_history()

        # eval_rewards, eval_steps = np.array([(row['Evaluated Reward'], row['_step']) for row in history if row['Evaluated Reward'] is not None]).T
        # Write the above line in a more readable way in a loop

        eval_rewards = []
        eval_steps = []
        counter = 0
        max_len = 99

        for i, row in enumerate(history):
            if row["Evaluated Reward"] is not None:
                print(counter, row["Evaluated Reward"], row["_step"])
                eval_rewards.append(row["Evaluated Reward"])
                eval_steps.append(row["_step"])
                counter += 1
                if counter == max_len:
                    break
            
        
        eval_rewards = np.array(eval_rewards)
        eval_steps = np.array(eval_steps)

        all_rewards.append(eval_rewards)
        all_steps.append(eval_steps)

    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    # max_len = len(all_rewards[0])

    avg_rewards = np.mean(all_rewards, axis=0).reshape(max_len)
    std_rewards = np.std(all_rewards, axis=0).reshape(max_len)
    steps = 1000 * np.arange(1, max_len + 1)

    np.save("avg_rewards_equi_contrastive_rl.npy", avg_rewards)
    np.save("std_rewards_equi_contrastive_rl.npy", std_rewards)
    np.save("steps_equi_contrastive_rl.npy", steps)


def generate_contrastive_curves_arrays(runs, cache=None, use_cache=True):
    all_rewards = []
    all_steps = []

    for x, run in enumerate(runs):
        configs = {k: v for k, v in run.config.items() if not k.startswith("_")}
        history = run.scan_history()

        # eval_rewards, eval_steps = np.array([(row['Evaluated Reward'], row['_step']) for row in history if row['Evaluated Reward'] is not None]).T
        # Write the above line in a more readable way in a loop

        eval_rewards = []
        eval_steps = []
        counter = 0
        max_len = 99
    
        for i,row in enumerate(history):
            if row['Evaluated Reward'] is not None:
                print(row['Evaluated Reward'], row['_step'])
                eval_rewards.append(row['Evaluated Reward'])
                eval_steps.append(row['_step'])
                counter += 1
                if counter == max_len:
                    break
        print()

        eval_rewards = np.array(eval_rewards)
        eval_steps = np.array(eval_steps)

        all_rewards.append(eval_rewards)
        all_steps.append(eval_steps)

    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    # max_len = len(all_rewards[0])
    

    avg_rewards = np.mean(all_rewards, axis=0).reshape(max_len)
    std_rewards = np.std(all_rewards, axis=0).reshape(max_len)
    steps = 1000 * np.arange(1, max_len + 1)

    np.save("avg_rewards_contrastive_rl.npy", avg_rewards)
    np.save("std_rewards_contrastive_rl.npy", std_rewards)
    np.save("steps_contrastive_rl.npy", steps)


def generate_full_contrastive_curves():
    avg_reward_contrastive = np.load("avg_rewards_contrastive_rl.npy")
    avg_reward_equi_contrastive = np.load("avg_rewards_equi_contrastive_rl.npy")

    all_steps_contrastive = np.load("steps_contrastive_rl.npy")
    all_steps_equi_contrastive = np.load("steps_equi_contrastive_rl.npy")

    std_reward_contrastive = np.load("std_rewards_contrastive_rl.npy")
    std_reward_equi_contrastive = np.load("std_rewards_equi_contrastive_rl.npy")

    f, ax = plt.subplots(1, 1, sharey=True, dpi=100)


    # ax.set_xticks([0,  100000,  200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    #               ['0', '100k', '200K','300K', '400K', '500K','600K','700K','800K','900K','1M'])

    ax.set_xticks([0,  100000,  200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000 ],
                  ['0', '100k', '200K','300K', '400K', '500K','600K','700K','800K','900K','1M','1.1M','1.2M','1.3M','1.4M','1.5M'])


    ax.plot(15 * all_steps_contrastive,avg_reward_contrastive,label="Equi Contrastive RL (Pooling)",)
    ax.fill_between(
        15 * all_steps_contrastive,
        avg_reward_contrastive - std_reward_contrastive,
        avg_reward_contrastive + std_reward_contrastive,
        alpha=0.3,
    )

    
    ax.plot(
        15 * all_steps_equi_contrastive,
        avg_reward_equi_contrastive,
        label="Equi Contrastive RL",
    )
    ax.fill_between(
        15 * all_steps_equi_contrastive,
        avg_reward_equi_contrastive - std_reward_equi_contrastive,
        avg_reward_equi_contrastive + std_reward_equi_contrastive,
        alpha=0.3,
    )


    # ax.set_xlim(0,1050000)
    ax.set_xlim(0, 1550000)
    # ax.set_xlim(0,550000)

    ax.set_ylim(0, 1)
    font = font_manager.FontProperties(size=12)
    leg = ax.legend(
        ncol=2,
        fancybox=True,
        bbox_to_anchor=((2 + 1) / 2, -0.37),
        loc="center",
        prop=font,
    )
    [line.set_linewidth(3.0) for line in leg.get_lines()]
    ax.grid(True, axis="y")
    ax.legend()

    f.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Gradient Steps", fontweight="semibold", fontsize=12)
    plt.ylabel("Discounted Returns", fontweight="semibold", fontsize=12)
    f.subplots_adjust(bottom=0.3)
    plt.legend()
    plt.savefig("plot.png")
    plt.imshow()




if __name__ == "__main__":
    api = wandb.Api()

    # # Fetch-Push
    # equi_run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/0z04p4yy")
    # equi_run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/ffz80c75")
    # equi_run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/kx57j741")
    # equi_run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/6zuggmd6")

    # run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/er04qug3")
    # run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/x3h8l9jr")
    # run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/4zyzxu7g")
    # run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/kzz4ybvy")



    # # Fetch-Pick-And-Place
    # equi_run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/4pu5z3eo")
    # equi_run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/zcqziq39")
    # equi_run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/311zmq1c")
    # equi_run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/9ym467ym")

    # run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/30cvzjj3")
    # run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/6il6o88k")
    # run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/3amturec")
    # run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/curw7m1e")

    # # Sawyer-Push
    # equi_run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/m4k03sxz")
    # equi_run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/ttwwgqh0")
    # equi_run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/rnk69hre")
    # equi_run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/kwhbgiv3")

    # run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/cyfvdh05")
    # run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/7uduyma8")
    # run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/sktxwrbf")
    # run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/shype05p")



    # # Fetch-Push Equi-Critic vs Fetch-Push Invar-Critic 
    # equi_run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/6rmsnhts")
    # equi_run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/d0svkxia")
    # equi_run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/e3ppytw3")
    # equi_run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/rlidaw98")

    # run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/3yavipf1")
    # run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/3yavipf1")
    # run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/mvdddwm2")
    # run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/tlcq34pz")


    # Fetch-Push Equi-Critic vs Sawyer-Push Invar-Critic 
    equi_run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/59zqjq0v")
    equi_run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/izqb43m4")
    equi_run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/di6183hd")
    # equi_run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/kwhbgiv3")

    run_1 = api.run("arsh-tangri2/Equi_Contrastive_RL/9wp4y01i")
    run_2 = api.run("arsh-tangri2/Equi_Contrastive_RL/7yvfs0bi")
    run_3 = api.run("arsh-tangri2/Equi_Contrastive_RL/w8v6qwch")
    # run_4 = api.run("arsh-tangri2/Equi_Contrastive_RL/vfp654z3")



    generate_equi_contrastive_curves_arrays([equi_run_1, equi_run_2, equi_run_3])
    generate_contrastive_curves_arrays([run_1, run_2, run_3])

    generate_full_contrastive_curves()
