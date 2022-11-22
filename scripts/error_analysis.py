import json
import numpy as np
import torch.nn.functional as F
import pickle
from tqdm import tqdm

orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup/preds.jsonl"
gtpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_gtpriors/preds.jsonl"
lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_lmpriors_gpt3/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_lmpriors_codex/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_lmpriors_codex/preds.jsonl"
randpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_randpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_lmgpriors/preds.jsonl"
orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_heldout_transition/preds.jsonl"
lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_heldout_transition_gpt3_priors/preds.jsonl"
task_num_to_desc = {23521: 'Make Jello Shots', 59684: 'Build Simple Floating Shelves', 71781: 'Make Taco Salad', 113766: 'Grill Steak', 105222: 'Make Kimchi Fried Rice', 94276: 'Make Meringue', 53193: 'Make a Latte', 105253: 'Make Bread and Butter Pickles', 44047: 'Make Lemonade', 76400: 'Make French Toast', 16815: 'Jack Up a Car', 95603: 'Make Kerala Fish Curry', 109972: 'Make Banana Ice Cream', 44789: 'Add Oil to Your Car', 40567: 'Change a Tire', 77721: 'Make Irish Coffee', 87706: 'Make French Strawberry Cake', 91515: 'Make Pancakes'}

gpt3_task_bigramprobs = np.load("saved_probabilities/lm_bigrams.pkl.npy", allow_pickle=True).item()

# orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup/preds.jsonl"
# gtpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_gtpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmgpriors/preds.jsonl"

# average over tasks
def get_intersection(fn, hardsplit_val_transition_videos):
    video_to_acc = {}
    task_to_acc = {}
    video_to_pred_seq = {}
    video_to_gt_seq = {}
    task_to_has_no_action_acc = {}
    task_to_videos = {}
    # has_action_acc = {}
    # no_action_acc = {}
    # hardsplit_val_transition_videos[video[1]]['no_action']
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
            if line['video'] in hardsplit_val_transition_videos[task_desc]['has_action']:
                task_to_has_no_action_acc[task_desc]['has_action'].append(intersection / total)
            if line['video'] in hardsplit_val_transition_videos[task_desc]['no_action']:
                task_to_has_no_action_acc[task_desc]['no_action'].append(intersection / total)
            video_to_pred_seq[(line["task"], task_desc, line["video"])] = line["pred"]
            video_to_gt_seq[(line["task"], task_desc, line["video"])] = line["actual"]
            video_to_acc[(line["task"], task_desc, line["video"])] = intersection / total
            if (line["task"], task_desc) not in task_to_acc:
                task_to_acc[(line["task"], task_desc)] = []
                task_to_videos[(line["task"], task_desc)] = []
            task_to_acc[(line["task"], task_desc)].append(intersection / total)
            task_to_videos[(line["task"], task_desc)].append(line["video"])
    for task in task_to_acc:
        task_to_acc[task] = sum(task_to_acc[task]) / len(task_to_acc[task])
    return video_to_acc, video_to_pred_seq, video_to_gt_seq, task_to_acc, task_to_has_no_action_acc, task_to_videos


hardsplit_val_transition_videos_fn = "data/crosstask/crosstask_release/videos_val_heldout_transition.jsonl"
hardsplit_val_transition_videos = {}
with open(hardsplit_val_transition_videos_fn) as f:
    for line in f:
        line = json.loads(line)
        hardsplit_val_transition_videos[line['task']] = line


task_num_to_step_model_order = {}
with open(f"expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup/train_init_transition_probs.jsonl") as f:
    all_lines = f.readlines()
    for line in tqdm(all_lines):
        line = json.loads(line)
        task = task_num_to_desc[line["task_num"]]
        actions = line["actions"]
        non_bkg_actions = [a for a in actions if a != "BKG"]
        task_num_to_step_model_order[line["task_num"]] = non_bkg_actions


def determine_has_transition(action_seq, transition):
    action1 = transition[0].replace(" ", "_")
    action2 = transition[1].replace(" ", "_")
    for i in range(len(action_seq)-1):
        if action_seq[i] == action1 and action_seq[i+1] == action2:
            return True
    return False


def compare_sort(video_to_acc, fn1, fn2, segmented_by_action_presence=None):
    diffs = {}
    fn1_acc_square = np.array([[0,0],[0,0]])
    fn2_acc_square = np.array([[0,0],[0,0]])
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
        # import pdb; pdb.set_trace()
        # skip task if not repaired by GPT3
        heldout_transition = hardsplit_val_transition_videos[video[1]]['transition']
        task_step_idxs = task_num_to_step_model_order[video[0]]
        task_step_tuples = (task_step_idxs.index(heldout_transition[0]), task_step_idxs.index(heldout_transition[1]))
        gpt3_probs = gpt3_task_bigramprobs[video[0]][task_step_tuples[1], task_step_tuples[0]]

        print(diffs[video], f"{video_to_acc[fn2][video]} - {video_to_acc[fn1][video]}", video, f"{gpt3_probs.item()*100:.2f}")
        if segmented_by_action_presence:
            print("    " + str(diffs_seg[video]['has_action']) + " // " + str(diffs_seg[video]['no_action']))
            # import pdb; pdb.set_trace()
            # if gpt3_probs > 0.5:   #and diffs_seg[video]['has_action'] < diffs_seg[video]['no_action']:
            for specific_video in task_to_videos[fn1][video]:
                # check if transition present
                has_transition = specific_video in hardsplit_val_transition_videos[video[1]]['has_action']
                # total_has_trans += has_transition
                fn1_has_trans = determine_has_transition(video_to_pred_seq[fn1][(video[0], video[1], specific_video)], heldout_transition)
                # old_has_trans += fn1_has_trans
                fn2_has_trans = determine_has_transition(video_to_pred_seq[fn2][(video[0], video[1], specific_video)], heldout_transition)
                # lmprior_has_trans += fn2_has_trans
                fn1_acc_square[int(has_transition), int(fn1_has_trans)] += 1
                fn2_acc_square[int(has_transition), int(fn2_has_trans)] += 1
                if has_transition and not fn2_has_trans:
                    import pdb; pdb.set_trace()
                """
                if has_transition:
                    if fn1_has_trans and not fn2_has_trans:
                        print(heldout_transition)
                        import pdb; pdb.set_trace()
                else:
                    fn1_has_trans = determine_has_transition(video_to_pred_seq[fn1][(video[0], video[1], specific_video)], heldout_transition)
                    old_has_trans += fn1_has_trans
                    fn2_has_trans = determine_has_transition(video_to_pred_seq[fn2][(video[0], video[1], specific_video)], heldout_transition)
                    lmprior_has_trans += fn2_has_trans
                    if fn1_has_trans and not fn2_has_trans:
                        print(heldout_transition)
                        import pdb; pdb.set_trace()
                # video_to_gt_seq[(video[0], video[1], specific_video)]
                """
    if segmented_by_action_presence:
        print(f"has action: {sum(has_action_diff_accs) / len(has_action_diff_accs)}")
        print(f"no action: {sum(no_action_diff_accs) / len(no_action_diff_accs)}")
        print(f"orig vids: {fn1_acc_square}")
        print(f"prior vids: {fn2_acc_square}")
    return sorted_videos


video_to_acc = {}
task_to_acc = {}
avg_accs = {}
video_to_pred_seq = {}
task_to_has_no_action_acc = {}
task_to_videos = {}
# on correct vs. incorrect
for fn in [orig_fn, gtpr_fn, lmpr_fn, randpr_fn]:
    print(fn)
    video_to_acc[fn], video_to_pred_seq[fn], video_to_gt_seq, task_to_acc[fn], task_to_has_no_action_acc[fn], task_to_videos[fn] = get_intersection(fn, hardsplit_val_transition_videos)
    avg_accs[fn] = sum(video_to_acc[fn].values()) / len(video_to_acc[fn])

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
compare_sort(task_to_acc, orig_fn, lmpr_fn, task_to_has_no_action_acc)
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

print(avg_accs[orig_fn], avg_accs[gtpr_fn], avg_accs[lmpr_fn], avg_accs[randpr_fn])
