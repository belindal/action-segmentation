import json
import os
import openai
import numpy as np
from retry import retry
from tqdm import tqdm
import torch.nn.functional as F
import torch
import copy
import random
import math
import matplotlib.pyplot as plt
import time


openai.api_key = os.getenv("OPENAI_API_KEY_JDA")
engine = "code-davinci-002"
gpt3_file = f"gpt3/cache_{engine}.jsonl"


def load_gpt3_cache(gpt3_file):
    os.makedirs(os.path.split(gpt3_file)[0], exist_ok=True)
    cache = {}
    if os.path.exists(gpt3_file):
        with open(gpt3_file) as f:
            all_cached_lines = f.readlines()
            for item in tqdm(all_cached_lines, desc="Loading GPT3 cache"):
                item = json.loads(item)
                cache[item['prompt']] = item['result']
    return cache
cache = load_gpt3_cache(gpt3_file)


def save_gpt3_result(gpt3_file, new_results):
    with open(gpt3_file, "a") as wf:
        for prompt in new_results:
            wf.write(json.dumps({"prompt": prompt, "result": new_results[prompt]}) + "\n")


@retry(openai.error.RateLimitError, delay=3, backoff=1.5, max_delay=30, tries=10)
def openai_completion_query(**kwargs):
    if engine.startswith("code"):
        time.sleep(3)
        print(f"query time: {time.time()}")
    return openai.Completion.create(**kwargs)


def gpt3_score_prompt(engine, input_prefix, classes, cache=None):
    new_cache_results = {}
    class_scores = []
    # optimal_class = None
    for cl in classes:
        input_str = input_prefix + cl
        if cache and input_str in cache:
            result = cache[input_str]
        else:
            result = openai_completion_query(
                engine=engine,
                prompt=input_str,
                max_tokens=0,
                logprobs=0,
                echo=True,
            )
            if engine.startswith("code"):
                print("Got result")
            # gpt3_output = result["choices"][0]
            new_cache_results[input_str] = result
            cache[input_str] = result
            save_gpt3_result(gpt3_file, {input_str: result})
        for token_position in range(len(result['choices'][0]['logprobs']['tokens'])):
            if ''.join(
                result['choices'][0]['logprobs']['tokens'][:token_position]
            ).strip() == input_prefix.strip():
                break
        score = sum(result['choices'][0]['logprobs']['token_logprobs'][token_position:]) #/ len(result['choices'][0]['logprobs']['token_logprobs'][token_position:])
        class_scores.append(score)
    return class_scores, new_cache_results


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

lm_init_probs_fn = "saved_probabilities/init/lm_bigrams.pkl.npy"
lm_global_init_probs_fn = "saved_probabilities/init/lm_globals.pkl.npy"
lm_bigram_transition_probs_fn = "saved_probabilities/transition/lm_bigrams.pkl.npy"
lm_global_transition_probs_fn = "saved_probabilities/transition/lm_globals.pkl.npy"
if os.path.exists(lm_global_init_probs_fn):
    lm_global_init_probs = np.load(lm_global_init_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_init_probs_fn):
    lm_init_probs = np.load(lm_init_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_bigram_transition_probs_fn):
    lm_bigram_transition_probs = np.load(lm_bigram_transition_probs_fn, allow_pickle=True).item()
if os.path.exists(lm_global_transition_probs_fn):
    lm_global_transition_probs = np.load(lm_global_transition_probs_fn, allow_pickle=True).item()
gpt3_input_demos = """Your task is to make gummy bears. Your actions are: add gelatin, add flavoring, pour water in saucepan, pour juice, pour mixture into cup, add pureed berries, stir mixture.
The first step is stir mixture: implausible.
The first step is pour water in saucepan: plausible.
The step after add pureed berries is stir mixture: plausible.
The step after add gelatin is add flavoring: plausible.
The step after stir mixture is pour water into saucepan: implausible.

Your task is to build a desk. Your actions are: screw desk, paint wood, sand wood, cut wood.
The step after screw desk is sand wood: implausible.
The first step is cut wood: plausible.
The step after sand wood is cut wood: implausible.
The step after sand wood is paint wood: plausible.
The first step is screw desk: implausible.
The step after cut wood is sand wood: plausible.

Your task is to make vegan french toast. Your actions are: dip bread into milk mixture, heat skillet, melt butter on skillet, mix egg replacer, mix maple syrup, mix milk, mix vanilla, place bread in skillet, remove toast from pan
The first step is dip bread into milk mixture: implausible.
The step after heat skillet is melt butter on skillet: plausible.
The step after place bread in skillet is melt butter on skillet: implausible.
The step after dip bread into milk mixture is place bread in skillet: plausible.
The step after remove toast from pan is heat skillet: implausible.
The first step is heat skillet: plausible.
The first step is melt button skillet: implausible.
The step after place bread in skillet is remove toast from pan: plausible.
"""

bigram_gpt3_input_demos = """Your task is to make gummy bears. Your set of actions is: {add gelatin, add flavoring, pour water in saucepan, pour juice, pour mixture into cup, add pureed berries, stir mixture}
The first step is pour water in saucepan
The step after add pureed berries is stir mixture
The step after add gelatin is add flavoring
The step after pour water in saucepan is add gelatin
The step after stir mixture is pour mixture into cup

Your task is to build a desk. Your set of actions is: {screw desk, paint wood, sand wood, cut wood}
The step after sand wood is paint wood
The first step is cut wood
The step after cut wood is sand wood
The step after paint wood is screw desk

Your task is to make vegan french toast. Your set of actions is: {remove toast from pan, mix vanilla, dip bread into milk mixture, melt butter on skillet, mix egg replacer, mix maple syrup, mix milk, place bread in skillet, heat skillet}
The step after heat skillet is melt butter on skillet
The step after dip bread into milk mixture is place bread in skillet
The first step is heat skillet
The step after place bread in skillet is remove toast from pan

"""

global_gpt3_input_demos = """Your task is to make gummy bears. Your set of actions is: {add gelatin, add flavoring, pour water in saucepan, pour juice, pour mixture into cup, add pureed berries, stir mixture}
The correct ordering is: pour water in saucepan, add gelatin, add flavoring, pour juice, add pureed berries, stir mixture, pour mixture into cup. 

Your task is to build a desk. Your set of actions is: {screw desk, paint wood, sand wood, cut wood}
The correct ordering is: cut wood, sand wood, paint wood, screw desk. 

Your task is to make vegan french toast. Your set of actions is: {remove toast from pan, mix vanilla, dip bread into milk mixture, melt butter on skillet, mix egg replacer, mix maple syrup, mix milk, place bread in skillet, heat skillet}
The correct ordering is: heat skillet, mix milk, mix egg replacer, mix maple syrup, mix vanilla, melt butter on skillet, dip bread into milk mixture, place bread in skillet, remove toast from pan

"""


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
            input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\n\nThe first step is ",
            classes=non_bkg_actions,
            cache=cache,
        )
        lm_init_probs[line["task_num"]] = F.softmax(torch.tensor(gpt3_scores), dim=0)
        lm_bigram_transition_probs[line["task_num"]] = []
        for action in tqdm(non_bkg_actions):
            # for next_action in non_bkg_actions:
            gpt3_scores, _ = gpt3_score_prompt(
                engine=engine,
                input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\n\nThe step after {action} is ",
                classes=non_bkg_actions,
                cache=cache,
            )
            lm_bigram_transition_probs[line["task_num"]].append(F.softmax(torch.tensor(gpt3_scores), dim=0))
        lm_bigram_transition_probs[line["task_num"]] = torch.stack(lm_bigram_transition_probs[line["task_num"]], dim=-1)

        # global probs
        non_bkg_actions_prompt = sorted(non_bkg_actions)
        # if task == "Add Oil to Your Car":
        gpt3_scores, _ = gpt3_score_prompt(
            engine=engine,
            # input_prefix=f"{gpt3_input_demos}Your task is to {task.lower()}. Your actions are: {', '.join(non_bkg_actions_prompt)}.\n\nThe first step is ",
            input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\nThe first step is ",
            classes=non_bkg_actions,
            cache=cache,
        )
        lm_bigram_transition_probs[line["task_num"]] = []
        for action in tqdm(non_bkg_actions):
            # for next_action in non_bkg_actions:
            gpt3_scores, _ = gpt3_score_prompt(
                engine=engine,
                input_prefix=f"{bigram_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\nThe step after {action} is ",
                classes=non_bkg_actions,
                cache=cache,
            )
            lm_bigram_transition_probs[line["task_num"]].append(F.softmax(torch.tensor(gpt3_scores), dim=0))
        lm_bigram_transition_probs[line["task_num"]] = torch.stack(lm_bigram_transition_probs[line["task_num"]], dim=-1)

        lm_global_transition_probs[line["task_num"]] = []
        curr_prompt = f"{global_gpt3_input_demos}Your task is to {task.lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\nThe correct ordering is: "
        for action_idx, action in enumerate(tqdm(non_bkg_actions)):
            # for next_action in non_bkg_actions:
            gpt3_scores, _ = gpt3_score_prompt(
                engine=engine,
                input_prefix=curr_prompt,
                classes=non_bkg_actions,
                cache=cache,
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

np.save("saved_probabilities/init/val.pkl", empirical_val_init_probs, allow_pickle=True)
np.save("saved_probabilities/transition/val.pkl", empirical_val_transition_probs, allow_pickle=True)
np.save("saved_probabilities/transition/train.pkl", empirical_transition_probs, allow_pickle=True)
np.save(lm_init_probs_fn, lm_init_probs, allow_pickle=True)
np.save(lm_global_init_probs_fn, lm_global_init_probs, allow_pickle=True)
np.save(lm_bigram_transition_probs_fn, lm_bigram_transition_probs, allow_pickle=True)
np.save(lm_global_transition_probs_fn, lm_global_transition_probs, allow_pickle=True)

def graph_probs(task_transition_probs, task_init_probs, task_labels):
    all_types = list(task_transition_probs.keys())
    for j, task in enumerate(tqdm(task_transition_probs[all_types[1]], desc="Making graphs")):
        max_num_plots = max([len(task_transition_probs[type][task]) for type in task_transition_probs if task in task_transition_probs[type]]) + 1
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
        os.makedirs(f"saved_probabilities/plots", exist_ok=True)
        fig.savefig(f"saved_probabilities/plots/{task_num_to_desc[task]}.png")
    # for task in task_transition_probs:

# empirical_val_transition_probs["init"] = empirical_val_init_probs
# lm_bigram_transition_probs["init"] = lm_init_probs
graph_probs(
    {"train": empirical_transition_probs, "val": empirical_val_transition_probs, "lm_bigram": lm_bigram_transition_probs, "lm_global": lm_global_transition_probs},
    {"val": empirical_val_init_probs, "lm_bigram": lm_init_probs, "lm_global": lm_global_init_probs},
    task_num_to_step_model_order
)