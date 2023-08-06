import json
import numpy as np
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import os


font = {'family' : 'normal'}
#         'weight' : 'bold',
#         'size'   : 22}
default_font = {}#"fontfamily": "Times New Roman"}
title_font = {
    **default_font,
    "fontsize": 40,
    "fontweight": "bold",
}
axes_label_font = {
    **default_font,
    "fontsize": 30,
    "fontweight": "bold",
}
axes_ticks_font = {
    **default_font,
    "fontsize": 30,
}
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])

matplotlib.rc('font', **font)
# matplotlib.rc('fontname', 'Times New Roman')
# plt.rcParams['fontname'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True

heldout_part = "heldout_transition"

setting = "ood"
if setting == "zs":
    orig_dir = "expts/crosstask_i3d-resnet-audio/paper/pca_semimarkov_sup_nobkg_no_priors/"
    lmpr_dir = "expts/crosstask_i3d-resnet-audio/paper/pca_semimarkov_sup_nobkg_gpt3_priors/"
elif setting == "ood":
    orig_dir = f"expts/crosstask_i3d-resnet-audio/paper/pca_semimarkov_sup_{heldout_part}"
    lmpr_dir = f"expts/crosstask_i3d-resnet-audio/paper/pca_semimarkov_sup_{heldout_part}_gpt3_priors_10"


orig_fn = os.path.join(orig_dir, "preds.jsonl")
lmpr_fn = os.path.join(lmpr_dir, "preds.jsonl")
orig_bigramprobs_fn = os.path.join(orig_dir, "train_init_transition_probs.jsonl")
lmpr_bigramprobs_fn = os.path.join(lmpr_dir, "train_init_transition_probs.jsonl")

# heldout_transition_mode = True
task_num_to_desc = {23521: 'Make Jello Shots', 59684: 'Build Simple Floating Shelves', 71781: 'Make Taco Salad', 113766: 'Grill Steak', 105222: 'Make Kimchi Fried Rice', 94276: 'Make Meringue', 53193: 'Make a Latte', 105253: 'Make Bread and Butter Pickles', 44047: 'Make Lemonade', 76400: 'Make French Toast', 16815: 'Jack Up a Car', 95603: 'Make Kerala Fish Curry', 109972: 'Make Banana Ice Cream', 44789: 'Add Oil to Your Car', 40567: 'Change a Tire', 77721: 'Make Irish Coffee', 87706: 'Make French Strawberry Cake', 91515: 'Make Pancakes'}

gpt3_task_bigramprobs = np.load("saved_probabilities/transition/lm_bigrams_text.pkl.npy", allow_pickle=True).item()
train_task_bigramprobs = {}
with open(orig_bigramprobs_fn) as f:
    for line in f:
        line = json.loads(line)
        train_task_bigramprobs[line["task_num"]] = np.exp(np.array(line["transition_probs"]))
# valid_task_bigramprobs = {}
# with open(f"expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_{heldout_part}/train_init_transition_probs.jsonl") as f:
#     for line in f:
#         line = json.loads(line)
#         valid_task_bigramprobs[line["task_num"]] = np.exp(np.array(line["transition_probs"]))
gpt3_smoothed_bigramprobs = {}
with open(lmpr_bigramprobs_fn) as f:
    for line in f:
        line = json.loads(line)
        gpt3_smoothed_bigramprobs[line["task_num"]] = np.exp(np.array(line["transition_probs"]))

# gt_task_bigramprobs = {}
# with open(f"expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_{heldout_part}_gt_priors/train_init_transition_probs.jsonl") as f:
#     for line in f:
#         line = json.loads(line)
#         gt_task_bigramprobs[line["task_num"]] = np.exp(np.array(line["transition_probs"]))

# np.load("saved_probabilities/train_init_transition_probs.jsonl", allow_pickle=True).item()
# orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup/preds.jsonl"
# gtpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_gtpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmgpriors/preds.jsonl"

# average over tasks
def get_intersection(fn, hardsplit_val_heldout_videos):
    video_to_acc = {}
    task_to_acc = {}
    task_action_to_acc = {}
    video_to_pred_seq = {}
    video_to_gt_seq = {}
    task_to_has_no_action_acc = {}
    task_to_videos = {}
    per_action_recalls = {}
    # has_action_acc = {}
    # no_action_acc = {}
    # hardsplit_val_heldout_videos[video[1]]['no_action']
    with open(fn) as f:
        for line in f:
            line = json.loads(line)
            if line["task"] == 71781: continue
            intersection = 0.0
            total = 0.0
            for ts, item in enumerate(line["pred"]):
                if item == line["actual"][ts]:
                    intersection += 1
                total += 1
            task_desc = task_num_to_desc[line["task"]]
            if task_desc not in task_to_has_no_action_acc:
                task_to_has_no_action_acc[task_desc] = {'has_action': [], 'no_action': []}
            if line['video'] in hardsplit_val_heldout_videos[task_desc]['has_action']:
                task_to_has_no_action_acc[task_desc]['has_action'].append(intersection / total)
            if line['video'] in hardsplit_val_heldout_videos[task_desc]['no_action']:
                task_to_has_no_action_acc[task_desc]['no_action'].append(intersection / total)
            video_to_pred_seq[(line["task"], task_desc, line["video"])] = line["pred"]
            video_to_gt_seq[(line["task"], task_desc, line["video"])] = line["actual"]
            video_to_acc[(line["task"], task_desc, line["video"])] = intersection / total
            if (line["task"], task_desc) not in task_to_acc:
                task_to_acc[(line["task"], task_desc)] = []
                task_to_videos[(line["task"], task_desc)] = []
            task_to_acc[(line["task"], task_desc)].append(intersection / total)
            task_to_videos[(line["task"], task_desc)].append(line["video"])
            # breakpoint()
            # actions = set(line["pred"]).union(set(line["actual"]))
            actions = set(line["actual"])
            # action recall
            for action in actions:
                # if (line["task"], action) not in task_action_to_acc:
                #     task_action_to_acc[(line["task"], action)] = []
                pred_indices = (np.array(line["pred"]) == action).nonzero()[0]
                if len(pred_indices) == 0:
                    continue
                if action not in per_action_recalls:
                    per_action_recalls[action] = [0,0]
                per_action_recalls[action][1] += 1
                try:
                    center_index = min(pred_indices, key=lambda x:abs(x-(pred_indices[0]+pred_indices[-1])/2))
                except:
                    breakpoint()
                if line["actual"][center_index] == action:
                    per_action_recalls[action][0] += 1
                # task_action_to_acc[(line["task"], action)] = intersection / total
    for task in task_to_acc:
        task_to_acc[task] = sum(task_to_acc[task]) / len(task_to_acc[task])
    return video_to_acc, video_to_pred_seq, video_to_gt_seq, task_to_acc, task_to_has_no_action_acc, task_to_videos, per_action_recalls


hardsplit_val_heldout_videos_fn = f"data/crosstask/crosstask_release/videos_val_{heldout_part}.jsonl"
hardsplit_val_heldout_videos = {}
with open(hardsplit_val_heldout_videos_fn) as f:
    for line in f:
        line = json.loads(line)
        hardsplit_val_heldout_videos[line['task']] = line


task_num_to_step_model_order = {}
with open(f"expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup/train_init_transition_probs.jsonl") as f:
    all_lines = f.readlines()
    for line in tqdm(all_lines):
        line = json.loads(line)
        task = task_num_to_desc[line["task_num"]]
        actions = line["actions"]
        non_bkg_actions = [a for a in actions if a != "BKG"]
        task_num_to_step_model_order[line["task_num"]] = non_bkg_actions


def determine_has_heldout(action_seq, heldout_item):
    if len(heldout_item) == 2:
        # is transition
        action1 = heldout_item[0].replace(" ", "_")
        action2 = heldout_item[1].replace(" ", "_")
        for i in range(len(action_seq)-1):
            if action_seq[i] == action1 and action_seq[i+1] == action2:
                return True
        return False
    else:
        # is action
        action = heldout_item[0].replace(" ", "_")
        return action in action_seq


def compare_sort(video_to_acc, fn1, fn2, segmented_by_action_presence=None):
    diffs = {}
    fn1_acc_square = np.array([[0,0],[0,0]])
    fn2_acc_square = np.array([[0,0],[0,0]])
    n_gpt3_right_priors = [0,0]
    # n_gt_right_priors = [0,0]
    # old_has_trans = 0
    # lmprior_has_trans = 0
    # total_has_trans = 0
    if segmented_by_action_presence:
        diffs_seg = {}
        has_action_diff_accs = []
        no_action_diff_accs = []
    for video in video_to_acc[fn1]:
        diffs[video] = video_to_acc[fn2][video] - video_to_acc[fn1][video]
        if segmented_by_action_presence:
            diffs_seg[video] = {}
            for k in segmented_by_action_presence[fn2][video[1]]:
                diffs_seg[video][k] = segmented_by_action_presence[fn2][video[1]][k] - segmented_by_action_presence[fn1][video[1]][k]
                # TODO print videos which got worse even though they had the heldout action/transition (maybe priors don't have the fix...)
                if k == "has_action":
                    has_action_diff_accs.append(diffs_seg[video][k])
                else:
                    no_action_diff_accs.append(diffs_seg[video][k])
    sorted_videos = sorted(list(video_to_acc[fn1].keys()), key=diffs.get, reverse=True)
    for video in sorted_videos:
        # skip task if not repaired by GPT3
        task_step_idxs = task_num_to_step_model_order[video[0]]
        if heldout_part == "heldout_transition":
            heldout_item = hardsplit_val_heldout_videos[video[1]]['transition']
            task_step_tuples = (task_step_idxs.index(heldout_item[0]), task_step_idxs.index(heldout_item[1]))
            gpt3_probs = gpt3_task_bigramprobs[video[0]][task_step_tuples[1], task_step_tuples[0]]
            # gt_smoothed_probs = gt_task_bigramprobs[video[0]][task_step_tuples[1], task_step_tuples[0]]
            orig_train_probs = train_task_bigramprobs[video[0]][task_step_tuples[1], task_step_tuples[0]]
            gpt3_smoothed_probs = gpt3_smoothed_bigramprobs[video[0]][task_step_tuples[1], task_step_tuples[0]]
            gpt3_prior_correct = gpt3_probs > 1. / len(gpt3_task_bigramprobs[video[0]])
            n_gpt3_right_priors[gpt3_prior_correct] += 1
            # import pdb; pdb.set_trace()
            # gt_prior_correct = gt_smoothed_probs > orig_train_probs  #gt_probs > 1. / len(gpt3_task_bigramprobs[video[0]])
            # n_gt_right_priors[gt_prior_correct] += 1
            # if gpt3_prior_correct:
            #     try:
            #         assert gpt3_smoothed_probs > orig_train_probs
            #     except:
            #         import pdb; pdb.set_trace()
            gpt3_prior_correct = True
        else:
            heldout_item = (hardsplit_val_heldout_videos[video[1]]['step'],)
            task_step = task_step_idxs.index(heldout_item[0])
            gpt3_probs = gpt3_task_bigramprobs[video[0]][task_step]
            orig_train_probs = train_task_bigramprobs[video[0]][task_step]
            gpt3_prior_correct = True
            # gt_prior_correct = True

        print(diffs[video], f"{video_to_acc[fn2][video]} - {video_to_acc[fn1][video]}", video) #, f"{gpt3_probs.item()*100:.2f}")
        if segmented_by_action_presence:
            print("    " + str(diffs_seg[video]['has_action']) + " // " + str(diffs_seg[video]['no_action']))
            if True:   #and diffs_seg[video]['has_action'] < diffs_seg[video]['no_action']:
                for specific_video in task_to_videos[fn1][video]:
                    # check if transition present
                    has_heldout = specific_video in hardsplit_val_heldout_videos[video[1]]['has_action']
                    # total_has_trans += has_heldout
                    fn1_has_trans = determine_has_heldout(video_to_pred_seq[fn1][(video[0], video[1], specific_video)], heldout_item)
                    # old_has_trans += fn1_has_trans
                    fn2_has_trans = determine_has_heldout(video_to_pred_seq[fn2][(video[0], video[1], specific_video)], heldout_item)
                    # lmprior_has_trans += fn2_has_trans
                    fn1_acc_square[int(has_heldout), int(fn1_has_trans)] += 1
                    fn2_acc_square[int(has_heldout), int(fn2_has_trans)] += 1
                    # has_preceding_action = 
                    # if has_heldout and not fn2_has_trans and fn1_has_trans:
                    #     print(heldout_item)
                    #     print(video_to_pred_seq[fn1][(video[0], video[1], specific_video)])
                    #     print(video_to_pred_seq[fn2][(video[0], video[1], specific_video)])
                    #     print(video_to_gt_seq[(video[0], video[1], specific_video)])
                    #     import pdb; pdb.set_trace()
                    """
                    if has_heldout:
                        if fn1_has_trans and not fn2_has_trans:
                            print(heldout_transition)
                            import pdb; pdb.set_trace()
                    else:
                        fn1_has_trans = determine_has_heldout(video_to_pred_seq[fn1][(video[0], video[1], specific_video)], heldout_item)
                        old_has_trans += fn1_has_trans
                        fn2_has_trans = determine_has_heldout(video_to_pred_seq[fn2][(video[0], video[1], specific_video)], heldout_item)
                        lmprior_has_trans += fn2_has_trans
                        if fn1_has_trans and not fn2_has_trans:
                            print(heldout_transition)
                            import pdb; pdb.set_trace()
                    # video_to_gt_seq[(video[0], video[1], specific_video)]
                    """
    if segmented_by_action_presence:
        print(f"has action: {sum(has_action_diff_accs) / len(has_action_diff_accs)}")
        print(f"no action: {sum(no_action_diff_accs) / len(no_action_diff_accs)}")
        print(f"orig vids: {fn1_acc_square} {(fn1_acc_square[0,0] + fn1_acc_square[1,1]) / fn1_acc_square.sum()}")
        print(f"prior vids: {fn2_acc_square} {(fn2_acc_square[0,0] + fn2_acc_square[1,1]) / fn2_acc_square.sum()}")
        print(f"# tasks with incorrect/correct gpt3 priors: {n_gpt3_right_priors}")
        # print(f"# tasks with incorrect/correct gt priors: {n_gt_right_priors}")
    return sorted_videos


def plot_objectwise_metrics(m1, m2, obj_order, save_fn):
    x = np.arange(len(m1))  # the label locations
    width = 0.75  # the width of the bars
    fig, ax = plt.subplots(figsize = (6, 2))

    m1_ious = np.array([m1[obj] for obj in obj_order]) * 100
    m2_ious = np.array([m2[obj] for obj in obj_order]) * 100
    mask = (m2_ious - m1_ious) > 0
    rects1 = ax.bar(x[mask], (m2_ious - m1_ious)[mask], width, color='green') #, label='Original')
    rects1 = ax.bar(x[~mask],( m2_ious - m1_ious)[~mask], width, color='red') #, label='Original')
    ax.axhline(y=0, color='k')
    ax.set(xticklabels=[])
    ax.tick_params(axis='y', labelsize=axes_ticks_font['fontsize'])
    fig.tight_layout()
    plt.savefig(save_fn)


video_to_acc = {}
task_to_acc = {}
avg_accs = {}
video_to_pred_seq = {}
task_to_has_no_action_acc = {}
task_to_videos = {}
per_action_recalls = {}
# on correct vs. incorrect
for fn in [orig_fn, lmpr_fn]:
    print(fn)
    video_to_acc[fn], video_to_pred_seq[fn], video_to_gt_seq, task_to_acc[fn], task_to_has_no_action_acc[fn], task_to_videos[fn], per_action_recalls[fn] = get_intersection(fn, hardsplit_val_heldout_videos)
    avg_accs[fn] = sum(task_to_acc[fn].values()) / len(task_to_acc[fn])

# compare_sort(task_to_acc, orig_fn, gtpr_fn)
print("\n====\n")
sorted_videos = compare_sort(video_to_acc, orig_fn, lmpr_fn)
print("\n====\n")
for fn in task_to_has_no_action_acc:
    for task in task_to_has_no_action_acc[fn]:
        for action_or_not in task_to_has_no_action_acc[fn][task]:
            try:
                task_to_has_no_action_acc[fn][task][action_or_not] = sum(task_to_has_no_action_acc[fn][task][action_or_not]) / len(task_to_has_no_action_acc[fn][task][action_or_not])
            except:
                import pdb; pdb.set_trace()
# compare_sort(task_to_has_no_action_acc, orig_fn, lmpr_fn)
sorted_tasks = compare_sort(task_to_acc, orig_fn, lmpr_fn, task_to_has_no_action_acc)
video = sorted_videos[-4]
video_to_pred_seq[lmpr_fn][video]
video_to_pred_seq[orig_fn][video]
video_to_gt_seq[video]
gt_init_priors = np.load("saved_probabilities/init/val.pkl.npy", allow_pickle=True).item()
gt_trans_priors = np.load("saved_probabilities/transition/val.pkl.npy", allow_pickle=True).item()
gpt3_init_priors = np.load("saved_probabilities/init/lm_bigrams_text.pkl.npy", allow_pickle=True).item()
gpt3_trans_priors = np.load("saved_probabilities/transition/lm_bigrams_text.pkl.npy", allow_pickle=True).item()

# compare_sort(task_to_acc, orig_fn, lmpr_fn)
# print("\n====\n")
# # for video in video_to_acc[orig_fn]:
# compare_sort(task_to_acc, randpr_fn, lmpr_fn)
#     print(video, video_to_acc[orig_fn][video], video_to_acc[gtpr_fn][video], video_to_acc[lmpr_fn][video])
# breakpoint()
# plot_objectwise_metrics(task_to_acc[orig_fn], task_to_acc[lmpr_fn], obj_order=sorted_tasks, save_fn="figures/vidact_{setting}_pertask.png")
plot_objectwise_metrics(task_to_acc[orig_fn], task_to_acc[lmpr_fn], obj_order=sorted_tasks, save_fn=f"figures/vidact_{setting}_pertask.png")
print(avg_accs[orig_fn], avg_accs[lmpr_fn])
for fn in per_action_recalls:
    for action in per_action_recalls[fn]:
        per_action_recalls[fn][action] = per_action_recalls[fn][action][0] / per_action_recalls[fn][action][1]
# breakpoint()
print(sum(per_action_recalls[orig_fn].values()) / len(per_action_recalls[orig_fn].values()), sum(per_action_recalls[lmpr_fn].values()) / len(per_action_recalls[lmpr_fn].values()))
