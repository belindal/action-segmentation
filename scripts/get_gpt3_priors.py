import json
import os
import openai
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
import copy
import random
import math
import matplotlib.pyplot as plt
import time
from scipy.special import kl_div
import argparse
import seaborn as sns
from scripts.gpt3_utils import load_gpt3_cache, save_gpt3_result, gpt3_score_prompt, global_gpt3_input_demos, bigram_gpt3_input_demos


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices=["text", "code", "random"], default="text", help="which GPT3 model to use")
args = parser.parse_args()

model_type = args.model_type
engine = f"{model_type}-davinci-002"
gpt3_file = f"gpt3/cache_{engine}.jsonl"
cache = load_gpt3_cache(gpt3_file)

task_num_to_desc = {}
task_num_to_step_val_order = {}
empirical_val_transition_probs = {}
empirical_val_init_probs = {}
ln = 0
with open("data/crosstask/crosstask_release/tasks_primary.txt") as f:
    for line in f:
        ln += 1
        if ln % 6 == 1:
            # next_line = f.readline().strip()
            if line.strip() == "": continue
            task_num = int(line.strip())
            task_desc = f.readline().strip()
            task_num_to_desc[task_num] = task_desc
            f.readline()
            f.readline()
            task_num_to_step_val_order[task_num] = f.readline().strip().split(',')
            empirical_val_transition_probs[task_num] = np.zeros([len(task_num_to_step_val_order[task_num]), len(task_num_to_step_val_order[task_num])])
            empirical_val_init_probs[task_num] = np.zeros(len(task_num_to_step_val_order[task_num]))
            ln += 4

task_num_to_step_model_order = {}
lm_init_probs = {}
lm_bigram_transition_probs = {}
lm_global_init_probs = {}
lm_global_transition_probs = {}
empirical_transition_probs = {}

lm_init_probs_fn = f"saved_probabilities/init/lm_bigrams_{args.model_type}.pkl.npy"
lm_global_init_probs_fn = f"saved_probabilities/init/lm_globals_{args.model_type}.pkl.npy"
lm_bigram_transition_probs_fn = f"saved_probabilities/transition/lm_bigrams_{args.model_type}.pkl.npy"
lm_global_transition_probs_fn = f"saved_probabilities/transition/lm_globals_{args.model_type}.pkl.npy"
if os.path.exists(lm_global_init_probs_fn):
    lm_global_init_probs = np.load(lm_global_init_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_init_probs_fn):
    lm_init_probs = np.load(lm_init_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_bigram_transition_probs_fn):
    lm_bigram_transition_probs = np.load(lm_bigram_transition_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_global_transition_probs_fn):
    lm_global_transition_probs = np.load(lm_global_transition_probs_fn, allow_pickle=True).item()



with open("empirical_transition_probs.jsonl") as f:
    all_lines = f.readlines()
    for line in tqdm(all_lines):
        line = json.loads(line)
        task = task_num_to_desc[line["task_num"]]
        actions = line["actions"]
        non_bkg_actions = [a for a in actions if a != "BKG"]
        task_num_to_step_model_order[line["task_num"]] = non_bkg_actions
        try:
            assert set(non_bkg_actions) == set(task_num_to_step_val_order[line["task_num"]])
        except:
            import pdb; pdb.set_trace()
        non_bkg_actions_idxs = [i for i,a in enumerate(actions) if a != "BKG"]
        trans_logp = np.array(line["task_probs"])
        trans_p = np.exp(trans_logp)
        nonbkg_trans_p = trans_p[np.array(non_bkg_actions_idxs)][:,np.array(non_bkg_actions_idxs)]
        renormalized_p = (nonbkg_trans_p / nonbkg_trans_p.sum(0))
        empirical_transition_probs[line["task_num"]] = renormalized_p

        # if not os.path.exists(lm_bigram_transition_probs_fn):
        non_bkg_actions_prompt = sorted(non_bkg_actions)
        # if task == "Add Oil to Your Car":
        gpt3_scores, _ = gpt3_score_prompt(
            engine=engine,
            # input_prefix=f"{gpt3_input_demos}Your task is to {task.lower()}. Your actions are: {', '.join(non_bkg_actions_prompt)}.\n\nThe first step is ",
            input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}. These actions may be out of order and your job is to order them.\nThe first step is ",
            classes=non_bkg_actions,
            cache=cache,
            gpt3_file=gpt3_file,
        )
        lm_init_probs[line["task_num"]] = F.softmax(torch.tensor(gpt3_scores), dim=0)
        lm_bigram_transition_probs[line["task_num"]] = []
        for action in tqdm(non_bkg_actions, desc="bigram"):
            # for next_action in non_bkg_actions:
            gpt3_scores, _ = gpt3_score_prompt(
                engine=engine,
                input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}. These actions may be out of order and your job is to order them.\nThe step after {action} is ",
                classes=non_bkg_actions,
                cache=cache,
                gpt3_file=gpt3_file,
            )
            lm_bigram_transition_probs[line["task_num"]].append(F.softmax(torch.tensor(gpt3_scores), dim=0))
        lm_bigram_transition_probs[line["task_num"]] = torch.stack(lm_bigram_transition_probs[line["task_num"]], dim=-1)

        # global probs
        lm_global_transition_probs[line["task_num"]] = []
        curr_prompt = f"{global_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\nThe correct ordering is: "
        for action_idx, action in enumerate(tqdm(non_bkg_actions, desc="global")):
            # for next_action in non_bkg_actions:
            gpt3_scores, _ = gpt3_score_prompt(
                engine=engine,
                input_prefix=curr_prompt,
                classes=non_bkg_actions,
                cache=cache,
                gpt3_file=gpt3_file,
            )
            best_action = non_bkg_actions[torch.tensor(gpt3_scores).argmax()]
            curr_prompt += best_action
            if action_idx == 0:
                lm_global_init_probs[line["task_num"]] = F.softmax(torch.tensor(gpt3_scores), dim=0)
            else:
                lm_global_transition_probs[line["task_num"]].append(F.softmax(torch.tensor(gpt3_scores), dim=0))
            if action_idx == len(non_bkg_actions):
                curr_prompt += "."
            else:
                curr_prompt += ", "
        # set last action
        lm_global_transition_probs[line["task_num"]].append(torch.zeros(len(non_bkg_actions)))
        lm_global_transition_probs[line["task_num"]] = torch.stack(lm_global_transition_probs[line["task_num"]], dim=-1)
        # import pdb; pdb.set_trace()

val_videos_fn = "data/crosstask/crosstask_release/videos_val.csv"
with open(val_videos_fn) as f:
    for line in f:
        datum = line.split(",")
        last_step = None
        curr_step = None
        with open(f"data/crosstask/crosstask_release/annotations/{datum[0]}_{datum[1]}.csv") as annot_f:
            for line in annot_f:
                if line.strip() == "": break
                last_step = curr_step
                curr_step = int(line.split(',')[0])-1
                # translate to model index
                curr_step = task_num_to_step_model_order[int(datum[0])].index(task_num_to_step_val_order[int(datum[0])][curr_step])
                if last_step is not None:
                    empirical_val_transition_probs[int(datum[0])][curr_step,last_step] += 1
                else:
                    empirical_val_init_probs[int(datum[0])][curr_step] += 1
        # import pdb; pdb.set_trace()

for task in empirical_val_transition_probs:
    empirical_val_transition_probs[task] = empirical_val_transition_probs[task] / empirical_val_transition_probs[task].sum(0)
    empirical_val_init_probs[task] = empirical_val_init_probs[task] / empirical_val_init_probs[task].sum(0)

np.save(f"saved_probabilities/init/val.pkl", empirical_val_init_probs, allow_pickle=True)
np.save(f"saved_probabilities/transition/val.pkl", empirical_val_transition_probs, allow_pickle=True)
np.save(f"saved_probabilities/transition/train.pkl", empirical_transition_probs, allow_pickle=True)
np.save(lm_init_probs_fn, lm_init_probs, allow_pickle=True)
np.save(lm_global_init_probs_fn, lm_global_init_probs, allow_pickle=True)
np.save(lm_bigram_transition_probs_fn, lm_bigram_transition_probs, allow_pickle=True)
np.save(lm_global_transition_probs_fn, lm_global_transition_probs, allow_pickle=True)

kl_divs_lm_val = {}
def graph_probs(task_transition_probs, task_init_probs, task_labels):
    all_types = list(task_transition_probs.keys())
    for j, task in enumerate(tqdm(task_transition_probs[all_types[1]], desc="Making graphs")):
        max_num_plots = max([len(task_transition_probs[type][task]) for type in task_transition_probs if task in task_transition_probs[type]]) + 1
        
        for type in all_types:
            fig, ax = plt.subplots()#figsize=(15,6))
            # import pdb; pdb.set_trace()
            # visualize and save
            differences = np.absolute(task_transition_probs[type][task] - np.transpose(task_transition_probs[type][task]))
            ax = sns.heatmap(differences, ax=ax) #, annot=True, fmt="d")
            # fig = ax.get_figure()
            plt.xticks(np.arange(len(task_labels[task]))+0.5, task_labels[task], fontsize=9, rotation=90)
            plt.yticks(np.arange(len(task_labels[task]))+0.5, task_labels[task], fontsize=9, rotation=0)
            fig.tight_layout()
            os.makedirs(f"saved_probabilities/plots_{args.model_type}/hm", exist_ok=True)
            fig.savefig(f"saved_probabilities/plots_{args.model_type}/hm/{task_num_to_desc[task]}.png")

        fig, axs = plt.subplots(
            max_num_plots,
            # len(task_transition_probs[all_types[0]]),
            # max([len(task_transition_probs[all_types[0]][task]) for task in task_transition_probs[all_types[0]]]),
            figsize=(5,5*max_num_plots),
        )
        for type in all_types:
            # plot init
            if type in task_init_probs:
                axs[0].plot(range(len(task_init_probs[type][task])), task_init_probs[type][task], label=type)
                axs[0].set_xticks(range(len(task_init_probs[type][task])))
                axs[0].set_xticklabels(task_labels[task], rotation=90)
                axs[0].set_title((task,"initial"))
                axs[0].legend()
            # task_transition_probs[type][task][:,0]
            for i in range(len(task_transition_probs[type][task])):
                try:
                    axs[i+1].plot(range(len(task_transition_probs[type][task][:,i])), task_transition_probs[type][task][:,i], label=type)
                    axs[i+1].set_xticks(range(len(task_transition_probs[type][task][:,i])))
                except:
                    import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                axs[i+1].set_xticklabels(task_labels[task], rotation=90)
                axs[i+1].set_title((task,task_labels[task][i]))
                axs[i+1].legend()
        fig.tight_layout()
        os.makedirs(f"saved_probabilities/plots_{args.model_type}", exist_ok=True)
        fig.savefig(f"saved_probabilities/plots_{args.model_type}/{task_num_to_desc[task]}.png")
        kl_divs_lm_val[task_num_to_desc[task]] = []
        kl_divs_lm_val[task_num_to_desc[task]].append(kl_div(task_init_probs["val"][task], task_init_probs["lm_bigram"][task]).sum())
        for i in range(len(task_transition_probs["val"][task])):
            if (task_transition_probs["val"][task][:,i] != task_transition_probs["val"][task][:,i]).any(): continue
            kl_divs_lm_val[task_num_to_desc[task]].append(kl_div(task_transition_probs["val"][task][:,i], task_transition_probs["lm_bigram"][task][:,i]).sum())
        kl_divs_lm_val[task_num_to_desc[task]] = torch.tensor(kl_divs_lm_val[task_num_to_desc[task]]).mean().item()
    print("KL Divs")
    print(kl_divs_lm_val)
    print(sum(kl_divs_lm_val.values()) / len(kl_divs_lm_val))
    # for task in task_transition_probs:

def merge_probs(probs_dict, task_num_to_step_model_order):
    """
    Merge task-specific probabilities into global logits across all tasks
    """
    # check if task steps are unique (straightforwardly merge)
    all_steps = []
    repeated_steps = []
    step_to_task = {}
    for task in task_num_to_step_model_order:
        for step in task_num_to_step_model_order[task]:
            if step in all_steps:
                repeated_steps.append((task, task_num_to_desc[task], step))
            if step not in step_to_task:
                step_to_task[step] = []
            step_to_task[step].append(task)
        all_steps.extend(task_num_to_step_model_order[task])

    task_pairs_to_common_steps = {}
    for task in task_num_to_step_model_order:
        for task2 in task_num_to_step_model_order:
            if task == task2: continue
            if len( set(task_num_to_step_model_order[task]).intersection(set(task_num_to_step_model_order[task2]))) > 0:
                task_pairs_to_common_steps[(task, task2)] = set(task_num_to_step_model_order[task]).intersection(set(task_num_to_step_model_order[task2]))
    # check if any 2 tasks share the 2 or more steps
    # 11 are repeated across multiple tasks: {'whisk mixture', 'add sugar', 'add onion', 'pour egg', 'add flour', 'pour water', 'brake on', 'pour milk', 'pour espresso', 'stir mixture', 'pour alcohol'}
    unique_repeated_steps = set([step[2] for step in repeated_steps])
    repeat_tasks = [step[1] for step in repeated_steps]
    import pdb; pdb.set_trace()
    repeated_step_to_tasks = {step_to_task[step] for step in unique_repeated_steps}
    assert len(all_steps) == len(set(all_steps))

    # merge init

    # merge transitions


# merge init_probs and transition_probs by task...
# merge_probs({
#     "init": lm_init_probs, "transition": lm_bigram_transition_probs,
# }, task_num_to_step_model_order)

# empirical_val_transition_probs["init"] = empirical_val_init_probs
# lm_bigram_transition_probs["init"] = lm_init_probs
graph_probs(
    {"train": empirical_transition_probs, "val": empirical_val_transition_probs, "lm_bigram": lm_bigram_transition_probs, "lm_global": lm_global_transition_probs},
    {"val": empirical_val_init_probs, "lm_bigram": lm_init_probs, "lm_global": lm_global_init_probs},
    task_num_to_step_model_order
)
