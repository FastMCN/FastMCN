# import


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*import][import:1]]
import torch
import numpy as np
from utils.utils import *
from torch.nn.functional import normalize
from torch.nn import Module

loss_func_set = Register()
# import:1 ends here

# v1

# $$center = mean(L2(ft_{i})) i \in C$$

# $$pos_loss = \sum_{i=1}^{i=c}(mean(1 - ft_{i}center_{i}^{T}))$$

# $$neg_loss = mean(\sum_{j=1}^{j=c}(max(0, ft_{i}center_{j}^{T}))) i, j \in C, i \neq j$$

# $$loss = neg_loss + pos_loss$$

# #+NAME: v1

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::v1][v1]]
@loss_func_set
def v1(output):
    output_len = len(output)
    output = [normalize(i) for i in output]
    mean = torch.stack([i.mean(0) for i in output]).t()
    cos_sim = [i.mm(mean) for i in output]
    pos_loss = [o.mm(o.t()) for o in output]
    pos_loss_p = [l < 0.99 for l in pos_loss]
    pos_loss = torch.stack([(l * p).sum() / (p.sum() + 1e-6)
                            for l, p in zip(pos_loss, pos_loss_p)])
    pos_loss = (1 - pos_loss).sum()
    acc = torch.cat([(cos_sim[i]).argmax(1) == i
                     for i in range(output_len)]).float().mean()
    cos_sim_p = [c > 0 for c in cos_sim]
    neg_loss = torch.stack([((cos_sim[i] * cos_sim_p[i]).sum(1) - cos_sim[i][:, i]).mean()
                            for i in range(output_len)])
    neg_loss_p = neg_loss > 0
    prob = neg_loss_p.float().mean()
    neg_loss = (neg_loss * neg_loss_p).mean()
    loss = pos_loss + neg_loss
    result = {"loss": loss,
              "pos_loss": pos_loss,
              "neg_loss": neg_loss,
              "prob": prob,
              "acc": acc}
    return result
# v1 ends here

# v1_2


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_2][v1_2:1]]
@loss_func_set
def v1_2(output, device):
    output_len = len(output)
    output = [normalize(i) for i in output]
    mean = torch.stack([i.mean(0) for i in output]).t()
    cos_sim = [i.mm(mean) for i in output]
    pos_sim = [o.mm(o.t()) for o in output]
    pos_sim_p = [l < 0.99 for l in pos_sim]
    pos_loss = torch.stack(
        [((1 - s) * p).sum() / (p.sum() + 1e-6) for s, p in zip(pos_sim, pos_sim_p)]
    ).sum()
    acc = (
        torch.cat([(cos_sim[i]).argmax(1) == i for i in range(output_len)])
        .float()
        .mean()
    )
    neg_loss = mean.t().mm(mean)
    neg_loss_p = (neg_loss > 0) * (torch.eye(neg_loss.shape[0]) == 0).to(device)
    neg_loss = (neg_loss * neg_loss_p).sum(1).mean()
    prob = neg_loss_p.float().mean()
    # neg_loss = (neg_loss * neg_loss_p).mean()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_2:1 ends here

# v1_3


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_3][v1_3:1]]
@loss_func_set
def v1_3(output, eye_mask, upper, eps):
    output_len = len(output)
    output = [normalize(i) for i in output]
    mean = torch.stack([i.mean(0) for i in output]).t()
    cos_sim = [i.mm(mean) for i in output]
    pos_sim = [o.mm(o.t()) for o in output]
    pos_sim_p = [l < upper for l in pos_sim]
    pos_loss = torch.stack(
        [((1 - s) * p).sum() / (p.sum() + eps) for s, p in zip(pos_sim, pos_sim_p)]
    ).sum()
    acc = (
        torch.cat([(cos_sim[i]).argmax(1) == i for i in range(len(output))])
        .float()
        .mean()
    )
    neg_loss = mean.t().mm(mean)
    neg_loss_p = (neg_loss > 0) * eye_mask[:neg_loss.shape[0], :neg_loss.shape[1]]
    neg_loss = (neg_loss * neg_loss_p).sum(1).mean()
    prob = neg_loss_p.float().mean()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_3:1 ends here

# v1_4


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_4][v1_4:1]]
@loss_func_set
def v1_4(output, eye_mask, upper, eps):
    output_len = len(output)
    output = [normalize(i) for i in output]
    mean = torch.stack([i.mean(0) for i in output]).t()
    cos_sim = [i.mm(mean) for i in output]
    pos_loss = [-2 * torch.log((1 + o.mm(o.t())) / 2) for o in output]
    pos_loss = torch.stack([l.sum() / ((l > eps).sum() + eps) for l in pos_loss])
    # pos_sim = torch.stack([o.mm(o.t()) for o in output])
    # pos_loss = - 2 * torch.log((1 + pos_sim) / 2)
    # pos_sim_min = -pos_sim.min().detach()
    # pos_sim_min = pos_sim_min * (pos_sim_min < 0)
    # pos_loss = -torch.log((pos_sim + pos_sim_min + eps) / (1 + pos_sim_min + eps))
    # pos_sim = [o.mm(o.t()) for o in output]
    # # pos_sim may be negative
    # pos_loss = torch.stack([-torch.log((pos_sim_min + s + eps) / (1 + pos_sim_min + eps))
    #                         for s in pos_sim])
    # pos_loss = pos_loss.sum([1, 2]) / (pos_loss > eps).sum([1, 2])
    pos_loss = pos_loss.sum()
    acc = (
        torch.cat([(cos_sim[i]).argmax(1) == i for i in range(len(output))])
        .float()
        .mean()
    )
    neg_loss = mean.t().mm(mean)
    neg_loss_p = (neg_loss > 0) * eye_mask[:neg_loss.shape[0], :neg_loss.shape[1]]
    neg_loss = -(torch.log((1 - neg_loss * neg_loss_p)).sum(1).mean())
    prob = neg_loss_p.float().mean()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result

# @loss_func_set
# def v1_4(output, eye_mask, upper, eps, pos_cons=1.0):
#     output_len = len(output)
#     output = [normalize(i) for i in output]
#     mean = torch.stack([i.mean(0) for i in output]).t()
#     cos_sim = [i.mm(mean) for i in output]
#     pos_sim = torch.stack([o.mm(o.t()) for o in output])
#     # pos_sim_min = -pos_sim.min().detach()
#     pos_sim_p = pos_sim > 0
#     pos_loss = ((pos_sim * 10).exp() * pos_sim_p.logical_not()) + pos_sim * pos_sim_p
#     pos_loss = -torch.log(pos_loss)
#     # pos_loss = -torch.log((pos_cons + pos_sim + eps) / (1 + pos_cons + eps))
#     # pos_sim = [o.mm(o.t()) for o in output]
#     # # pos_sim may be negative
#     # pos_loss = torch.stack([-torch.log((pos_sim_min + s + eps) / (1 + pos_sim_min + eps))
#     #                         for s in pos_sim])
#     pos_loss = pos_loss.sum([1, 2]) / (pos_loss > eps).sum([1, 2])
#     pos_loss = pos_loss.sum()
#     acc = (
#         torch.cat([(cos_sim[i]).argmax(1) == i for i in range(len(output))])
#         .float()
#         .mean()
#     )
#     neg_loss = mean.t().mm(mean)
#     neg_loss_p = (neg_loss > 0) * eye_mask[:neg_loss.shape[0], :neg_loss.shape[1]]
#     neg_loss = -(torch.log((1 - neg_loss * neg_loss_p)).sum(1).mean())
#     prob = neg_loss_p.float().mean()
#     loss = pos_loss + neg_loss
#     result = {
#         "loss": loss,
#         "pos_loss": pos_loss,
#         "neg_loss": neg_loss,
#         "prob": prob,
#         "acc": acc,
#     }
#     return result
# v1_4:1 ends here

# v1_5


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_5][v1_5:1]]
@loss_func_set
def v1_5(output, eye_mask, upper, eps, pos_loss_times=2.0):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_loss = - pos_loss_times * torch.log((1 + output_mat_sim * concept_mask_not + concept_mask) / 2).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_loss = -torch.log(1 - (output_mat_sim * neg_mask)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_5:1 ends here

# v1_6

# The original version is the best. pow(2) and times 2 can't be better.


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_6][v1_6:1]]
@loss_func_set
def v1_6(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score + eps)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result

@loss_func_set
def v1_6_times_2(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - 2 * ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_6:1 ends here

# v1_6_release


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_6_release][v1_6_release:1]]
@loss_func_set
def v1_6_release(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    is_neg = torch.stack([label != i for i in label])
    is_pos = torch.logical_not(is_neg)
    pos_score = output_mat_sim * is_pos + is_neg + eps
    pos_score = pos_score * (pos_score < 1) + (pos_score >= 1)
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (is_pos.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = output_mat_sim - eps
    neg_p = is_neg * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_p
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / (neg_p.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_p.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_6_release:1 ends here

# v1_6

# The original version is the best. pow(2) and times 2 can't be better.


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_6][v1_6:1]]
@loss_func_set
def v1_6_neg1(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask
    neg_score = output_mat_sim * neg_mask
    neg_loss = - ((1 + neg_score) * torch.log((1 - neg_score) / 2 + eps)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result

@loss_func_set
def v1_6_times_2(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - 2 * ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_6:1 ends here

# v1_6_fast


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_6_fast][v1_6_fast:1]]
@loss_func_set
def v1_6_fast(output, n_names, eye_mask, eps):
    output_len = len(n_names)
    output_mat = normalize(output_mat, dim=-1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat_sim.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in n_names]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o, device="cuda") * l
            for l, o in zip(range(output_len), n_names)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score + eps)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_6_fast:1 ends here

# v1_6_jit


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_6_jit][v1_6_jit:1]]
@torch.jit.script
def v1_6_jit_pos_loss(output_mat_sim, concept_mask, concept_mask_not, eps):
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    return pos_loss


@torch.jit.script
def v1_6_jit_acc(output_mat_sim, eye_mask, label_range, label):
    pred = (output_mat_sim * eye_mask).argmax(1)
    pred = (pred > label_range).sum(0)
    acc = (pred == label).float().mean()
    return acc


@torch.jit.script
def v1_6_jit_neg_loss(output_mat_sim, concept_mask, eps):
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score)
                  ).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    return prob, neg_loss


@torch.jit.script
def v1_6_jit_eval(output_mat, eye_mask, label, label_range, concept_mask, eps):
    output_mat_sim = output_mat.mm(output_mat.t())
    concept_mask_not = torch.logical_not(concept_mask)
    acc = v1_6_jit_acc(output_mat_sim, eye_mask, label_range, label)
    pos_loss = v1_6_jit_pos_loss(
        output_mat_sim, concept_mask, concept_mask_not, eps)
    prob, neg_loss = v1_6_jit_neg_loss(output_mat_sim, concept_mask, eps)
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc
    }
    return result


@loss_func_set
def v1_6_jit(output, eye_mask, eps):
    # variables
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    dt_size = output_mat.shape[0]
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    concept_mask = torch.stack([label != i for i in label])
    eye_mask = eye_mask[:dt_size, :dt_size]
    result = v1_6_jit_eval(output_mat, eye_mask, label,
                           label_range, concept_mask, eps)
    return result
# v1_6_jit:1 ends here

# v1_7

# pow(2) is better.

# | type   | step |    acc |
# |--------+------+--------|
# | no pow |    1 |  0.032 |
# |        |    2 |  0.056 |
# |        |    3 | 0.0658 |
# | pow    |    1 |  0.028 |
# |        |    2 |  0.063 |
# |        |    3 |  0.082 |
# |        |    4 |  0.099 |
# |        |    5 |  0.112 |
# |        |    6 |  0.121 |
# |        |    7 |  0.129 |
# |        |    8 |  0.136 |
# |        |    9 |  0.144 |
# |        |   10 |  0.153 |
# |        |   11 |  0.160 |
# |        |   12 |  0.167 |
# |        |   13 |  0.173 |



# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_7][v1_7:1]]
@loss_func_set
def v1_7(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score_p = torch.clamp_min(output_mat_sim * concept_mask_not, 0)
    pos_score_p = pos_score_p + (pos_score_p == 0).float()
    pos_loss_p = - ((1 - pos_score_p) * torch.log(pos_score_p)).sum(0)
    pos_score_n = torch.clamp_max(output_mat_sim * concept_mask_not, 0)
    pos_score_n = pos_score_n + (pos_score_n == 0).float()
    pos_loss_n = - ((1 - pos_score_n) * torch.log((1 + pos_score_n) / 2)).sum(0)
    pos_loss = pos_loss_p + pos_loss_n
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v1_7:1 ends here

# v1_8


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_8][v1_8:1]]
@loss_func_set
def v1_8(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = (output_mat_sim * concept_mask).pow(2)
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / concept_mask.sum(0)
    neg_loss = neg_loss.sum()
    # prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        # "prob": prob,
        "acc": acc,
    }
    return result

@loss_func_set
def v1_8_neg_times_2(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = (output_mat_sim * concept_mask).pow(2)
    neg_loss = - 2 * (neg_score * torch.log(1 - neg_score)).sum(0) / concept_mask.sum(0)
    neg_loss = neg_loss.sum()
    # prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        # "prob": prob,
        "acc": acc,
    }
    return result

@loss_func_set
def v1_8_abs(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = (output_mat_sim * concept_mask).abs()
    neg_loss = - (neg_score * torch.log(1 - neg_score)).sum(0) / concept_mask.sum(0)
    neg_loss = neg_loss.sum()
    # prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        # "prob": prob,
        "acc": acc,
    }
    return result

@loss_func_set
def v1_8_log(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) -1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = output_mat_sim * concept_mask
    neg_loss = (- neg_score * torch.log(1 - neg_score)).sum(0) / concept_mask.sum(0)
    neg_loss = neg_loss.sum()
    # prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        # "prob": prob,
        "acc": acc,
    }
    return result
# v1_8:1 ends here

# v1_9


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_9][v1_9:1]]
@loss_func_set
def v1_9(output, eye_mask, eps, neg_freq, pos_freq):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    # pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_score = torch.sin((pos_score - 1) * pos_freq)
    pos_loss = - (0.5 * (2 - pos_score) * torch.log(1 + pos_score)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = output_mat_sim * concept_mask
    neg_score = torch.cos_(neg_score * neg_freq)
    neg_loss = - (0.5 * (2 - neg_score) * torch.log((1 + neg_score) / 2)).sum(0) / (concept_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    loss = pos_loss + neg_loss

    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "acc": acc,
    }
    return result


@loss_func_set
def v1_9_pos_sin_2(output, eye_mask, eps, neg_freq, pos_freq):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    # pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_score = torch.sin(pos_score * pos_freq)
    pos_loss = - ((2 + pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = output_mat_sim * concept_mask
    neg_score = torch.cos_(neg_score * neg_freq)
    neg_loss = - (0.5 * (2 - neg_score) * torch.log((1 + neg_score) / 2)).sum(0) / (concept_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "acc": acc
    }
    return result

@loss_func_set
def v1_9_pos_log(output, eye_mask, eps, neg_freq, pos_freq):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_score = output_mat_sim * concept_mask
    neg_score = torch.cos_(neg_score * neg_freq)
    neg_loss = - (0.5 * (2 - neg_score) * torch.log((1 + neg_score) / 2)).sum(0) / (concept_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "acc": acc,
    }
    return result


@loss_func_set
def v1_9_pos_log_neg_no_neg(output, eye_mask, eps, neg_freq, pos_freq):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_score = output_mat_sim * concept_mask_not + concept_mask
    pos_loss = - ((1 - pos_score) * torch.log((1 + pos_score) / 2)).sum(0)
    pos_loss = pos_loss / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_score = output_mat_sim * neg_mask
    neg_score = torch.cos_(neg_score * neg_freq)
    neg_loss = - (0.5 * (2 - neg_score) * torch.log((1 + neg_score) / 2)).sum(0) / (neg_mask.sum(0) + eps)
    neg_loss = neg_loss.sum()
    prob = neg_mask.float().mean()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "acc": acc,
        "prob": prob
    }
    return result
# v1_9:1 ends here

# v1_10


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v1_10][v1_10:1]]
@loss_func_set
def v1_10(output, eye_mask, freq, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).abs().argmax(1)
    label_range = (
        torch.from_numpy(np.cumsum([i.shape[0] for i in output]))
        .to("cuda")
        .repeat([dt_size, 1])
        .t()
    )
    pred = (pred > label_range).sum(0)
    label = torch.cat(
        [
            torch.ones(o.shape[0], device="cuda") * l
            for l, o in zip(range(output_len), output)
        ]
    )
    acc = (pred == label).float().mean()
    # concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    score = torch.cos(output_mat_sim * freq)
    # score = score.clamp_min(-1 + 1e-6)
    # loss = - (0.5 * (2 - score) * torch.log((1 + score) / 2)).mean(0).sum()
    pos_loss = (score * concept_mask_not + 1) / (concept_mask_not.sum(0) - 1 + eps)
    pos_loss = pos_loss.sum()
    neg_loss = (- score * concept_mask + 1).sum(0) / concept_mask.sum(0)
    neg_loss = neg_loss.sum()
    loss = pos_loss + neg_loss
    result = {
        "loss": loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "acc": acc
    }
    return result
# v1_10:1 ends here

# v2


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v2][v2:1]]
# @loss_func_set
# def v2(output, threshold=1.0, eps=1e-6):
#     output_len = len(output)
#     output = [normalize(i) for i in output]
#     mean = torch.cat([i.mean(0).unsqueeze(1) for i in output], 1)
#     cos_sim = [i.mm(mean) for i in output]
#     acc = (
#         torch.cat([(cos_sim[i]).argmax(1) == i for i in range(output_len)])
#         .float()
#         .mean()
#     )
#     intra_cos_sim = [
#         c[:, i].repeat([c.shape[1], 1]).t() for c, i in zip(cos_sim, range(output_len))
#     ]
#     cos_sim_delta = [c * (c != i) - i for c, i in zip(cos_sim, intra_cos_sim)]
#     pos_loss = [1 + cos_sim_delta[i][:, i] for i in range(output_len)]
#     pos_loss = torch.cat(pos_loss)
#     pos_loss_p = pos_loss > eps
#     pos_loss = (pos_loss * pos_loss_p).sum() / pos_loss_p.sum()
#     loss = [threshold + d for d in cos_sim_delta]
#     loss_p = [l > eps for l in loss]
#     loss = [(l * p).sum(1) / (p.sum(1) + 1e-6) for l, p in zip(loss, loss_p)]
#     loss = torch.cat(loss).mean() + pos_loss
#     loss = torch.cat(loss).mean()
#     rank = torch.cat([(d > 0).float().mean(1) for d in cos_sim_delta]).mean()
#     rank = torch.cat([r.float().mean(1) for r in rank_p]).mean()
#     assert not torch.isnan(loss), f"loss is nan.\n{[p.sum(1) for p in loss_p]}"
#     result = {"loss": loss,
#               "acc": acc,
#               "pos_loss": pos_loss,
#               "rank": rank}
#     return result


@loss_func_set
def v2(output, threshold=1.0, eps=1e-6):
    output_len = len(output)
    output = [normalize(i) for i in output]
    mean = torch.cat([i.mean(0).unsqueeze(1) for i in output], 1)
    cos_sim = [i.mm(mean).t() for i in output]
    acc = (
        torch.cat([(cos_sim[i]).argmax(0) == i for i in range(output_len)])
        .float()
        .mean()
    )
    intra_cos_sim = [c[i]
                     for c, i in zip(cos_sim, range(output_len))]
    cos_sim_delta = [c * (c != i) - i for c, i in zip(cos_sim, intra_cos_sim)]
    # pos_loss = [1 + cos_sim_delta[i][i] for i in range(output_len)]
    # pos_loss = torch.cat(pos_loss)
    # pos_loss_p = pos_loss > eps
    # pos_loss = (pos_loss * pos_loss_p).sum() / pos_loss_p.sum()
    loss = [threshold + d for d in cos_sim_delta]
    loss_p = [l > 1e-6 for l in loss]
    loss = [(l * p).sum(0) / (p.sum(0) + 1e-6) for l, p in zip(loss, loss_p)]
    # loss = torch.cat(loss).mean() + pos_loss
    loss = torch.cat(loss).mean()
    rank = torch.cat([(d > 0) for d in cos_sim_delta], -1).float().mean()
    assert not torch.isnan(loss), f"loss is nan.\n{[p.sum(1) for p in loss_p]}"
    result = {"loss": loss,
              "acc": acc,
              # "pos_loss": pos_loss,
              "rank": rank}
    return result
# v2:1 ends here

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v2][v2:2]]
@loss_func_set
def v2_2(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = torch.from_numpy(np.cumsum([i.shape[0] for i in output])).to("cuda").repeat([dt_size, 1]).t()
    pred = (pred > label_range).sum(0)
    label = torch.cat([torch.ones(o.shape[0], device="cuda") * l
                       for l, o in zip(range(output_len), output)])
    acc = (pred == label).float().mean()
    concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_sim = (output_mat_sim * concept_mask_not).sum(0) - 1
    pos_sim = pos_sim / (concept_mask_not.sum(0) - 1 + eps)
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_sim = (output_mat_sim * neg_mask).sum(0) / (neg_mask.sum(0) + eps)
    loss = 1 - pos_sim + neg_sim
    loss = loss.mean()
    prob = neg_mask.float().mean()
    # loss_p = loss > 0
    # prob = loss_p.float().mean()
    # loss = (loss * loss_p).sum()

    result = {
        "loss": loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v2:2 ends here

# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/loss_func.org::*v2][v2:3]]
@loss_func_set
def v2_3(output, eye_mask, eps):
    output_len = len(output)
    output_mat = torch.cat(output)
    output_mat = normalize(output_mat, dim=1)
    output_mat_sim = output_mat.mm(output_mat.t())
    dt_size = output_mat.shape[0]
    pred = (output_mat_sim * eye_mask[:dt_size, :dt_size]).argmax(1)
    label_range = torch.from_numpy(np.cumsum([i.shape[0] for i in output])).to("cuda").repeat([dt_size, 1]).t()
    pred = (pred > label_range).sum(0)
    label = torch.cat([torch.ones(o.shape[0], device="cuda") * l
                       for l, o in zip(range(output_len), output)])
    acc = (pred == label).float().mean()
    concept_mask = torch.stack([label != i for i in label]).float()
    concept_mask = torch.stack([label != i for i in label])
    concept_mask_not = torch.logical_not(concept_mask)
    pos_sim = (output_mat_sim * concept_mask_not).sum(0) - 1
    pos_sim = pos_sim / (concept_mask_not.sum(0) - 1 + eps)
    neg_mask = concept_mask * (output_mat_sim > 0)
    neg_sim = (output_mat_sim * neg_mask).sum(0) / (neg_mask.sum(0) + eps)
    loss = (1 + pos_sim - neg_sim) / 2
    loss = loss * (loss > 0) + (loss < 0)
    loss = loss.mean()
    prob = neg_mask.float().mean()
    # loss_p = loss > 0
    # prob = loss_p.float().mean()
    # loss = (loss * loss_p).sum()
    result = {
        "loss": loss,
        "prob": prob,
        "acc": acc,
    }
    return result
# v2:3 ends here
