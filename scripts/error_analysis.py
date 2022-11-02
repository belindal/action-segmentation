import json

orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg/preds.jsonl"
gtpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_gtpriors/preds.jsonl"
lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_lmpriors/preds.jsonl"
# lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_lmgpriors/preds.jsonl"
task_num_to_desc = {23521: 'Make Jello Shots', 59684: 'Build Simple Floating Shelves', 71781: 'Make Taco Salad', 113766: 'Grill Steak', 105222: 'Make Kimchi Fried Rice', 94276: 'Make Meringue', 53193: 'Make a Latte', 105253: 'Make Bread and Butter Pickles', 44047: 'Make Lemonade', 76400: 'Make French Toast', 16815: 'Jack Up a Car', 95603: 'Make Kerala Fish Curry', 109972: 'Make Banana Ice Cream', 44789: 'Add Oil to Your Car', 40567: 'Change a Tire', 77721: 'Make Irish Coffee', 87706: 'Make French Strawberry Cake', 91515: 'Make Pancakes'}

orig_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup/preds.jsonl"
gtpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_gtpriors/preds.jsonl"
lmpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmpriors/preds.jsonl"
# lmgpr_fn = "expts/crosstask_i3d-resnet-audio/pca_semimarkov_unsup_lmgpriors/preds.jsonl"

# average over tasks
def get_intersection(fn):
    video_to_acc = {}
    task_to_acc = {}
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
            video_to_acc[(line["task"], task_num_to_desc[line["task"]], line["video"])] = intersection / total
            if (line["task"], task_num_to_desc[line["task"]]) not in task_to_acc:
                task_to_acc[(line["task"], task_num_to_desc[line["task"]])] = []
            task_to_acc[(line["task"], task_num_to_desc[line["task"]])].append(intersection / total)
    for task in task_to_acc:
        task_to_acc[task] = sum(task_to_acc[task]) / len(task_to_acc[task])
    return video_to_acc, task_to_acc


def compare_sort(video_to_acc, fn1, fn2):
    diffs = {}
    for video in video_to_acc[fn1]:
        diffs[video] = video_to_acc[fn2][video] - video_to_acc[fn1][video]
    sorted_videos = sorted(list(video_to_acc[fn1].keys()), key=diffs.get, reverse=True)
    for video in sorted_videos:
        print(video, diffs[video], f"{video_to_acc[fn2][video]} - {video_to_acc[fn1][video]}")


video_to_acc = {}
task_to_acc = {}
avg_accs = {}
for fn in [orig_fn, gtpr_fn, lmpr_fn]:
    video_to_acc[fn], task_to_acc[fn] = get_intersection(fn)
    avg_accs[fn] = sum(video_to_acc[fn].values()) / len(video_to_acc[fn])

compare_sort(task_to_acc, orig_fn, gtpr_fn)
print("\n====\n")
# compare_sort(video_to_acc, orig_fn, lmpr_fn)
compare_sort(task_to_acc, orig_fn, lmpr_fn)
# for video in video_to_acc[orig_fn]:
#     print(video, video_to_acc[orig_fn][video], video_to_acc[gtpr_fn][video], video_to_acc[lmpr_fn][video])

print(avg_accs[orig_fn], avg_accs[gtpr_fn], avg_accs[lmpr_fn])
