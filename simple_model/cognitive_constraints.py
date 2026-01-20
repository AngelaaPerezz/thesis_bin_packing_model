import numpy as np


global MEMORY_LIMIT
MEMORY_LIMIT = 4


def apply_planning_constraint(items):
    # Get unplaced items with their areas and memory limit
    areas = []
    ids = []

    for item in items:
        if item[3] == -1:  # not placed yet
            area = item[1] * item[2] # height * width
            areas.append(area)
            ids.append(item[0])  # item_id 

    if len(ids) > 0:
        k = min(MEMORY_LIMIT, len(ids))
        items = np.array(ids)[np.argsort(areas)[-k:]]
    else:
        items = np.array([])
    return items

def get_candidate_actions(obs, item_id, max_candidates=5):
    """
    Returns up to max_candidates valid actions for a given item_id,
    upper-top first, near already placed items.
    """
    valid_indices = np.where(obs['action_mask'] == 1)[0]
    valid_actions = obs['obs']['actions'][valid_indices]

    # keep only actions for this item
    item_actions = valid_actions[valid_actions[:, 0] == item_id]

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
