"""
Make hard train splits
"""
import json
import os
from gpt3_utils import task_num_to_desc, task_num_to_step_val_order


all_videos_fn = "data/crosstask/crosstask_release/videos.csv"
val_videos_fn = "data/crosstask/crosstask_release/videos_val.csv"

hardsplit_train_step_videos_fn = "data/crosstask/crosstask_release/videos_train_heldout_step.csv"
hardsplit_train_transition_videos_fn = "data/crosstask/crosstask_release/videos_train_heldout_transition.csv"

val_tasks = set()
val_videos = set()
with open(val_videos_fn) as f:
    for line in f:
        task = line.split(",")[0]
        vid_id = line.split(",")[1]
        val_tasks.add(int(task))
        val_videos.add(vid_id)


def make_heldout_step_train_splits(videos_fn):
    # vid_to_missing_steps = {}
    task_to_steps_to_count = {}
    split_to_task_to_steps_to_vids = {"train": {}, "valid": {}}
    split_to_task_to_vids = {"train": {}, "valid": {}}
    with open(videos_fn) as f:
        for line in f:
            datum = line.split(",")
            task = int(datum[0])
            vid_id = datum[1]
            if task not in val_tasks:
                continue
            if vid_id in val_videos:
                split = "valid"
            else:
                split = "train"
            if task not in task_to_steps_to_count:
                task_to_steps_to_count[task] = {}
            if task not in split_to_task_to_steps_to_vids[split]:
                split_to_task_to_steps_to_vids[split][task] = {}
                split_to_task_to_vids[split][task] = []
            split_to_task_to_vids[split][task].append(vid_id)
            steps = set()
            with open(f"data/crosstask/crosstask_release/annotations/{datum[0]}_{datum[1]}.csv") as annot_f:
                for line in annot_f:
                    if line.strip() == "": break
                    curr_step = int(line.split(',')[0])-1
                    step_nl = task_num_to_step_val_order[task][curr_step]
                    steps.add(step_nl)
                    # if step_nl not in task_to_steps_to_count[task]:
                    #     task_to_steps_to_count[task][step_nl] = 0
                    # task_to_steps_to_count[task][step_nl] += 1
            for step_nl in steps:
                if step_nl not in task_to_steps_to_count[task]:
                    task_to_steps_to_count[task][step_nl] = 0
                task_to_steps_to_count[task][step_nl] += 1
                if step_nl not in split_to_task_to_steps_to_vids[split][task]:
                    split_to_task_to_steps_to_vids[split][task][step_nl] = []
                split_to_task_to_steps_to_vids[split][task][step_nl].append(vid_id)
            # import pdb; pdb.set_trace()
    task_to_actionwithleasttrainvids = {}
    task_to_step_to_percentvidspresent = {}
    persplit_task_to_step_to_percentvidspresent = {"train": {}, "valid": {}}
    persplit_task_to_step_to_remainingvids = {"train": {}, "valid": {}}
    # for split in split_to_task_to_steps_to_vids:
    for task in split_to_task_to_steps_to_vids["train"]:
        persplit_task_to_step_to_percentvidspresent["train"][task] = {}
        persplit_task_to_step_to_percentvidspresent["valid"][task] = {}
        persplit_task_to_step_to_remainingvids["train"][task] = {}
        persplit_task_to_step_to_remainingvids["valid"][task] = {}
        task_to_actionwithleasttrainvids[task] = None
        for split in split_to_task_to_steps_to_vids:
            for step in task_to_steps_to_count[task]:
                persplit_task_to_step_to_percentvidspresent[split][task][step] = len(split_to_task_to_steps_to_vids[split][task][step]) / len(split_to_task_to_vids[split][task])
                persplit_task_to_step_to_remainingvids[split][task][step] = len(split_to_task_to_vids[split][task]) - len(split_to_task_to_steps_to_vids[split][task][step])
        for action in split_to_task_to_steps_to_vids["train"][task]:
            # import pdb; pdb.set_trace()
            if task_to_actionwithleasttrainvids[task] is None:
                task_to_actionwithleasttrainvids[task] = action
            elif persplit_task_to_step_to_remainingvids["train"][task][action] > persplit_task_to_step_to_remainingvids["train"][task][task_to_actionwithleasttrainvids[task]]:
                task_to_actionwithleasttrainvids[task] = action
        print(task_num_to_desc[task])
        # print(split_to_task_to_steps_to_vids["train"][task])
        # print(split_to_task_to_steps_to_vids["valid"][task])
        print(f"    {task_to_actionwithleasttrainvids[task]} {persplit_task_to_step_to_percentvidspresent['train'][task][task_to_actionwithleasttrainvids[task]]} ({persplit_task_to_step_to_remainingvids['train'][task][task_to_actionwithleasttrainvids[task]]} remaining)")
        print(f"    {task_to_actionwithleasttrainvids[task]} {persplit_task_to_step_to_percentvidspresent['valid'][task][task_to_actionwithleasttrainvids[task]]}")
        # print(persplit_task_to_step_to_percentvidspresent["valid"][task])
    with open(hardsplit_train_step_videos_fn, "w") as wf:
        for task in split_to_task_to_vids["train"]:
            for video in split_to_task_to_vids["train"][task]:
                if video in split_to_task_to_steps_to_vids["train"][task][task_to_actionwithleasttrainvids[task]]:
                    continue
                line = f"{task},{video},https://www.youtube.com/watch?v={video}"
                wf.write(line+"\n")
    # for task in empirical_val_transition_probs:
    #     empirical_val_transition_probs[task] = empirical_val_transition_probs[task] / empirical_val_transition_probs[task].sum(0)
    #     empirical_val_init_probs[task] = empirical_val_init_probs[task] / empirical_val_init_probs[task].sum(0)
    return split_to_task_to_vids


def make_heldout_transition_train_splits(videos_fn):
    split_to_task_to_transition_to_vids = {"train": {}, "valid": {}}
    split_to_task_to_vids = {"train": {}, "valid": {}}
    with open(videos_fn) as f:
        for line in f:
            datum = line.split(",")
            last_step = None
            curr_step = None
            task = int(datum[0])
            vid_id = datum[1]
            if task not in val_tasks:
                continue
            if vid_id in val_videos:
                split = "valid"
            else:
                split = "train"
            if task not in split_to_task_to_transition_to_vids[split]:
                split_to_task_to_transition_to_vids[split][task] = {}
                split_to_task_to_vids[split][task] = []
            split_to_task_to_vids[split][task].append(vid_id)
            transitions = set()
            with open(f"data/crosstask/crosstask_release/annotations/{datum[0]}_{datum[1]}.csv") as annot_f:
                for line in annot_f:
                    if line.strip() == "": break
                    last_step = curr_step
                    curr_step = task_num_to_step_val_order[task][int(line.split(',')[0])-1]
                    if last_step is not None:
                        transition = (last_step, curr_step)
                        transitions.add(transition)
            for trans in transitions:
                if trans not in split_to_task_to_transition_to_vids[split][task]:
                    split_to_task_to_transition_to_vids[split][task][(trans)] = []
                split_to_task_to_transition_to_vids[split][task][trans].append(vid_id)
    task_to_transitionwithmostvalidvids = {}
    task_to_transition_to_percentvidspresent = {}
    persplit_task_to_transition_to_percentvidspresent = {"train": {}, "valid": {}}
    persplit_task_to_transition_to_remainingvids = {"train": {}, "valid": {}}
    for task in split_to_task_to_transition_to_vids["train"]:
        persplit_task_to_transition_to_percentvidspresent["train"][task] = {}
        persplit_task_to_transition_to_percentvidspresent["valid"][task] = {}
        persplit_task_to_transition_to_remainingvids["train"][task] = {}
        persplit_task_to_transition_to_remainingvids["valid"][task] = {}
        task_to_transitionwithmostvalidvids[task] = None
        for split in split_to_task_to_transition_to_vids:
            for trans in split_to_task_to_transition_to_vids[split][task]:
                persplit_task_to_transition_to_percentvidspresent[split][task][trans] = len(split_to_task_to_transition_to_vids[split][task][trans]) / len(split_to_task_to_vids[split][task])
                persplit_task_to_transition_to_remainingvids[split][task][trans] = len(split_to_task_to_vids[split][task]) - len(split_to_task_to_transition_to_vids[split][task][trans])
        for trans in split_to_task_to_transition_to_vids["train"][task]:
            # import pdb; pdb.set_trace()
            if task_to_transitionwithmostvalidvids[task] is None:
                task_to_transitionwithmostvalidvids[task] = trans
            elif persplit_task_to_transition_to_remainingvids["train"][task].get(trans, 0) > max(85, 0.3*len(split_to_task_to_vids[split][task])) and persplit_task_to_transition_to_percentvidspresent["valid"][task].get(trans, 0) > persplit_task_to_transition_to_percentvidspresent["valid"][task].get(task_to_transitionwithmostvalidvids[task], 0):
                task_to_transitionwithmostvalidvids[task] = trans
            elif persplit_task_to_transition_to_remainingvids["train"][task].get(trans, 0) > persplit_task_to_transition_to_remainingvids["train"][task].get(task_to_transitionwithmostvalidvids[task], 0) and persplit_task_to_transition_to_percentvidspresent["valid"][task].get(trans, 0) == persplit_task_to_transition_to_percentvidspresent["valid"][task].get(task_to_transitionwithmostvalidvids[task], 0):
                task_to_transitionwithmostvalidvids[task] = trans
        print(task_num_to_desc[task])
        # print(persplit_task_to_transition_to_percentvidspresent["valid"][task])
        # print(persplit_task_to_transition_to_remainingvids["train"][task])
        print(f"    {task_to_transitionwithmostvalidvids[task]} {persplit_task_to_transition_to_percentvidspresent['train'][task][task_to_transitionwithmostvalidvids[task]]} ({persplit_task_to_transition_to_remainingvids['train'][task][task_to_transitionwithmostvalidvids[task]]} remaining)")
        print(f"    {task_to_transitionwithmostvalidvids[task]} {persplit_task_to_transition_to_percentvidspresent['valid'][task].get(task_to_transitionwithmostvalidvids[task], 0)}")
    with open(hardsplit_train_transition_videos_fn, "w") as wf:
        for task in split_to_task_to_vids["train"]:
            for video in split_to_task_to_vids["train"][task]:
                if video in split_to_task_to_transition_to_vids["train"][task][task_to_transitionwithmostvalidvids[task]]:
                    continue
                line = f"{task},{video},https://www.youtube.com/watch?v={video}"
                wf.write(line+"\n")


split_to_task_to_vids = make_heldout_step_train_splits(all_videos_fn)
print("\n")
make_heldout_transition_train_splits(all_videos_fn)
# get_split_info(val_videos_fn)
