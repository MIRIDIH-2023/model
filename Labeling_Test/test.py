import torch
import random

def random_masking(L=4096, mask_ratio=0.75):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(L)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=0)

    # keep the first subset
    ids_keep = ids_shuffle[:len_keep]
    ids_remove = ids_shuffle[len_keep:]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([L])
    mask[:len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=0, index=ids_restore)
    print(f'mask = {mask}')
    return mask, ids_restore, ids_remove, ids_keep

# Argument : random_masking의 mask
# Returns : masking할 곳의 slice들
def group_tokens(mask):

    group_lst = []
    i=0
    prev=0

    for m in mask:
        if m == 0:
            if i == prev:
                pass
            else:
                group_lst.append([prev, i])
            prev = i+1
        i += 1

    if prev != i:
        group_lst.append([prev, i])

    print(f'group list : {group_lst}')
    return group_lst

# Argument : ori_bbox_lst, group_tokens의 리턴 list (slices)
# Returns : masking된 부분의 그룹화된 bbox
def group_bbox(bbox_lst, group_lst):

    bbox_group_lst = []

    for s in group_lst:
        target = bbox_lst[s[0]:s[1]]
        if len(target) == 1:
            bbox_group_lst.append(*target)
        else:
            t = target[0][0]
            l = target[0][1]
            b = target[0][2]
            r = target[0][3]
            for i in target[1:]:
                if i[0] < t:
                    t = i[0]
                if i[1] < l:
                    l = i[1]
                if i[2] > b:
                    b = i[2]
                if i[3] > r:
                    b = i[3]
            bbox_group_lst.append([t,l,b,r])
    
    return bbox_group_lst

# Argument : ori_bbox_list, mask_ratio
# Returns : token slices to be masked, grouped bboxes
def mask_process(bbox_list, mask_ratio=0.75):
    l = len(bbox_list)
    mask = random_masking(L=l, mask_ratio=mask_ratio)
    return group_tokens(mask[0]), group_bbox(bbox_list, group_tokens(mask[0]))


L = 10

bbox_lst = [[random.randint(0,1280), random.randint(0,720), random.randint(0,1280), random.randint(0,720)] for i in range(L)]

print(bbox_lst)

for i in range(10):
    print(mask_process(bbox_lst))
    print()