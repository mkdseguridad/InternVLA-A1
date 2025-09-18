def convert_action_abs2delta(data, action_keys):
    """
    Convert absolute action to delta action.
    Args:
        data: dict, data to convert
        action_keys: list, action keys to convert
    Returns:
        data: dict, data with delta action
    """
    for action_key in action_keys:
        if action_key not in data:
            raise ValueError(f"Action key {action_key} not found in data")

        delta_action = data[action_key].diff(dim=0)
        data[action_key] = delta_action
        data[f"{action_key}_is_pad"] = data[f"{action_key}_is_pad"][1:]

    return data

def convert_action_abs2rel(data, action_keys):
    """
    Convert absolute action to relative action.
    """
    for action_key in action_keys:
        if action_key not in data:
            raise ValueError(f"Action key {action_key} not found in data")

        rel_action = data[action_key] - data[action_key][0]
        data[action_key] = rel_action[1:]
        data[f"{action_key}_is_pad"] = data[f"{action_key}_is_pad"][1:]

    return data