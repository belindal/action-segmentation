import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scripts.gpt3_utils import gpt3_score_prompt, load_gpt3_cache, global_gpt3_input_demos, task_num_to_desc

# model_type = args.model_type
model_type = "code"
engine = f"{model_type}-davinci-002"
gpt3_file = f"gpt3/cache_{engine}.jsonl"
gpt3_cache = load_gpt3_cache(gpt3_file)

def labels_to_spans(position_labels, max_k):
    # position_labels: b x N, LongTensor
    assert not (position_labels == -1).any(), "position_labels already appear span encoded (have -1)"
    b, N = position_labels.size()
    last = position_labels[:, 0]
    values = [last.unsqueeze(1)]
    lengths = torch.ones_like(last)
    for n in range(1, N):
        this = position_labels[:, n]
        same_symbol = (last == this)
        if max_k is not None:
            same_symbol = same_symbol & (lengths < max_k - 1)
        encoded = torch.where(same_symbol, torch.full([1], -1, device=same_symbol.device, dtype=torch.long), this)
        lengths = torch.where(same_symbol, lengths, torch.full([1], 0, device=same_symbol.device, dtype=torch.long))
        lengths += 1
        values.append(encoded.unsqueeze(1))
        last = this
    return torch.cat(values, dim=1)


def rle_spans(spans, lengths):
    b, T = spans.size()
    all_rle = []
    for i in range(b):
        this_rle = []
        this_spans = spans[i, :lengths[i]]
        current_symbol = None
        count = 0
        for symbol in this_spans:
            symbol = symbol.item()
            if current_symbol is None or symbol != -1:
                if current_symbol is not None:
                    assert count > 0
                    this_rle.append((current_symbol, count))
                count = 0
                current_symbol = symbol
            count += 1
        if current_symbol is not None:
            assert count > 0
            this_rle.append((current_symbol, count))
        assert sum(count for sym, count in this_rle) == lengths[i]
        all_rle.append(this_rle)
    return all_rle


def spans_to_labels(spans):
    # spans: b x N, LongTensor
    # contains 0.. for the start of a span (B-*), and -1 for its continuation (I-*)
    b, N = spans.size()
    current_labels = spans[:, 0]
    assert (current_labels != -1).all()
    values = [current_labels.unsqueeze(1)]
    for n in range(1, N):
        this = spans[:, n]
        this_labels = torch.where(this == -1, current_labels, this)
        values.append(this_labels.unsqueeze(1))
        current_labels = this_labels
    return torch.cat(values, dim=1)


def get_diagonal_covariances(data):
    # data: num_points x feat_dim
    model = GaussianMixture(n_components=1, covariance_type='diag')
    responsibilities = np.ones((data.shape[0], 1))
    model._initialize(data, responsibilities)
    return model.covariances_, model.precisions_cholesky_


def semimarkov_sufficient_stats(feature_list, label_list, covariance_type, n_classes, max_k=None):
    assert len(feature_list) == len(label_list)
    tied_diag = covariance_type == 'tied_diag'
    if tied_diag:
        emissions = GaussianMixture(n_classes, covariance_type='diag')
    else:
        emissions = GaussianMixture(n_classes, covariance_type=covariance_type)
    X_l = []
    r_l = []

    span_counts = np.zeros(n_classes, dtype=np.float32)
    span_lengths = np.zeros(n_classes, dtype=np.float32)
    span_start_counts = np.zeros(n_classes, dtype=np.float32)
    # to, from
    span_transition_counts = np.zeros((n_classes, n_classes), dtype=np.float32)

    instance_count = 0

    # for i in tqdm.tqdm(list(range(len(train_data))), ncols=80):
    for X, labels in zip(feature_list, label_list):
        X_l.append(X)
        r = np.zeros((X.shape[0], n_classes))
        r[np.arange(X.shape[0]), labels] = 1
        assert r.sum() == X.shape[0]
        r_l.append(r)
        spans = labels_to_spans(labels.unsqueeze(0), max_k)
        # symbol, length
        spans = rle_spans(spans, torch.LongTensor([spans.size(1)]))[0]
        last_symbol = None
        for index, (symbol, length) in enumerate(spans):
            if index == 0:
                span_start_counts[symbol] += 1
            span_counts[symbol] += 1
            span_lengths[symbol] += length
            if last_symbol is not None:
                span_transition_counts[symbol, last_symbol] += 1
            last_symbol = symbol
        instance_count += 1

    X_arr = np.vstack(X_l)
    r_arr = np.vstack(r_l)
    # import pdb; pdb.set_trace()
    emissions._initialize(X_arr, r_arr)
    if tied_diag:
        cov, prec_chol = get_diagonal_covariances(X_arr)
        emissions.covariances_[:] = np.copy(cov)
        emissions.precisions_cholesky_[:] = np.copy(prec_chol)
    return emissions, {
        'span_counts': span_counts,
        'span_lengths': span_lengths,
        'span_start_counts': span_start_counts,
        'span_transition_counts': span_transition_counts,
        'instance_count': instance_count,
    }


def make_steps_lm_prefix(actions_list):
    steps_nl = []
    for i, action in enumerate(actions_list):
        steps_nl.append(f"{i}. {action}")
    return ", ".join(steps_nl) + f"{len(actions_list)}. "


def greedy_lm_search(scores, task, lengths, add_eos, valid_classes_mapping, corpus_index2label):
    bs = len(lengths)
    pred_spans = -torch.ones([bs, max(lengths)], device=scores['emission'].device, dtype=torch.long)
    actions_uniq_history = [[] for _ in lengths]
    curr_action_lens = torch.zeros(bs, device=scores['emission'].device, dtype=torch.long)
    valid_classes = [valid_class for valid_class in list(valid_classes_mapping.keys()) if valid_classes_mapping[valid_class] not in [-1, 106]]
    non_bkg_actions_prompt = [corpus_index2label[valid_classes_mapping[valid_class]] for valid_class in valid_classes]
    # [[] for _ in lengths]
    for i in range(max(lengths)):
        next_action_logprob = torch.zeros([bs, len(valid_classes)], device=scores['emission'].device)
        # emission probabilities
        next_action_logprob += scores['emission'][:,i]

        # lm probabilities
        next_action_logprob_lm = []
        for b in range(len(lengths)):
            steps_until_now = make_steps_lm_prefix(actions_uniq_history[b])
            curr_prompt = f"{global_gpt3_input_demos}Your task is to {task_num_to_desc[task].lower()}. Your set of actions is: {{{', '.join(non_bkg_actions_prompt)}}}\nThe steps are: {steps_until_now}"
            # LM probabilities
            # TODO exclude last action??? (or include LM probability...)
            next_action_logprob_lm.append(gpt3_score_prompt(
                engine=engine,
                input_prefix=curr_prompt,
                classes=non_bkg_actions_prompt,
                cache=gpt3_cache,
                gpt3_file=gpt3_file,
            )[0])
        import pdb; pdb.set_trace()
        next_action_logprob += torch.tensor(next_action_logprob_lm, device=scores['emission'].device)

        # length biases
        if i > 0:
            # by batch
            pred_spans[i]
    # for 
    return pred_spans
