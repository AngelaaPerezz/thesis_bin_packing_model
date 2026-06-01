import numpy as np


global MEMORY_LIMIT
MEMORY_LIMIT = 4


def apply_planning_constraint(items):

     # filter unplaced items
    unplaced_items = [item for item in items if item[3] == -1]

    if len(unplaced_items) == 0:
        return []

    # sort unplaced items 
    sorted_items = sorted(unplaced_items, key=lambda x: x[1]*x[2], reverse=True)

    # apply memory limit and get the top k greatest items
    k = min(MEMORY_LIMIT, len(sorted_items))
    top_items = sorted_items[:k]

    return top_items



def get_candidate_actions(obs, item_id, max_candidates=5):
    """
    Returns up to max_candidates valid actions for a given item_id,
    upper-top first, near already placed items.
    """
    valid_indices = np.where(obs['action_mask'] == 1)[0]
    valid_actions = obs['obs']['actions'][valid_indices]

    # keep only actions for this item
    item_actions = valid_actions[valid_actions[:, 0] == item_id[0] ]

    if len(item_actions) == 0:
        return []

    # top-down filling
    item_actions_sorted = item_actions[
        np.lexsort((
            item_actions[:, 1],   # x  (ascendente)
            -item_actions[:, 2]   # y  (descendente)
        ))
    ]    

    # Return up to max_candidates
    return item_actions_sorted[:max_candidates]
