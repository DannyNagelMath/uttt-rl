import numpy as np

# Observation vector layout (109 elements):
#   [0:81]   board_flat         (81 values, {-1, 0, 1})
#   [81:90]  sub_board_winner   (9 values,  {-1, 0, 1})
#   [90:99]  sub_board_drawn    (9 values,  {0, 1})      -- symmetric
#   [99]     current_player     (1 value,   {-1, 1})
#   [100:109] active_board mask (9 values,  {0, 1})      -- symmetric


def flip_obs(obs: np.ndarray) -> np.ndarray:
    """
    Flip an observation so that the agent always perceives itself as X (1).

    Used when the agent is actually playing as O (-1), either during
    self-play training or human-vs-agent play.

    Transformations applied:
        board_flat       — negated  (swaps X and O marks)
        sub_board_winner — negated  (swaps X and O ownership)
        sub_board_drawn  — unchanged (draw is symmetric)
        current_player   — set to 1 (agent always thinks it is X)
        active_board     — unchanged (board geometry is symmetric)

    Parameters
    ----------
    obs : np.ndarray, shape (109,), dtype int8
        Observation vector as produced by UTTTEnv._get_obs().

    Returns
    -------
    np.ndarray, shape (109,), dtype int8
        Flipped observation.
    """
    flipped = obs.copy()
    flipped[0:81]  = -obs[0:81]    # board_flat
    flipped[81:90] = -obs[81:90]   # sub_board_winner
    flipped[99]    = 1              # current_player
    return flipped