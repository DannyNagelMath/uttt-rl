import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from cnn_env import CNNEnv


def test_obs_shape_and_dtype():
    """Observation has correct shape and dtype after reset."""
    env = CNNEnv()
    obs, _ = env.reset()
    assert obs.shape == (6, 9, 9), f"Expected (6,9,9), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    print("PASS test_obs_shape_and_dtype")


def test_initial_obs_values():
    """At game start: no pieces, all 81 cells legal, no sub-board outcomes."""
    env = CNNEnv()
    obs, _ = env.reset()

    assert np.all(obs[0] == 0), "ch0 (my pieces) should be empty at start"
    assert np.all(obs[1] == 0), "ch1 (opponent pieces) should be empty at start"
    assert np.all(obs[2] == 1), "ch2 (legal moves) should be all 1 at start — free choice"
    assert np.all(obs[3] == 0), "ch3 (my won sub-boards) should be empty at start"
    assert np.all(obs[4] == 0), "ch4 (opponent won sub-boards) should be empty at start"
    assert np.all(obs[5] == 0), "ch5 (drawn sub-boards) should be empty at start"
    print("PASS test_initial_obs_values")


def test_board_reshape_correctness():
    """Piece at (br, bc, lr, lc) appears at grid position (br*3+lr, bc*3+lc)."""
    env = CNNEnv()
    env.reset()

    # Place X at (br=0, bc=1, lr=2, lc=0) — expect grid position (2, 3)
    env.game.board[0, 1, 2, 0] = 1
    env.game.current_player = 1
    obs = env._get_obs()
    assert obs[0, 2, 3] == 1.0, f"Expected piece at grid (2,3), ch0={obs[0]}"

    # Place O at (br=2, bc=2, lr=1, lc=1) — expect grid position (7, 7)
    env.game.board[2, 2, 1, 1] = -1
    env.game.current_player = -1
    obs = env._get_obs()
    assert obs[0, 7, 7] == 1.0, f"Expected piece at grid (7,7) in ch0, ch0={obs[0]}"

    print("PASS test_board_reshape_correctness")


def test_perspective_channels():
    """ch0 always shows current player's pieces; ch1 shows opponent's."""
    env = CNNEnv()
    env.reset()

    # Place X at (0,0,0,0), O at (0,1,0,0)
    env.game.board[0, 0, 0, 0] = 1   # X at grid (0, 0)
    env.game.board[0, 1, 0, 0] = -1  # O at grid (0, 3)

    # From X's perspective
    env.game.current_player = 1
    obs = env._get_obs()
    assert obs[0, 0, 0] == 1.0, "X's piece should be in ch0 when X is current player"
    assert obs[1, 0, 3] == 1.0, "O's piece should be in ch1 when X is current player"

    # From O's perspective — channels should swap
    env.game.current_player = -1
    obs = env._get_obs()
    assert obs[0, 0, 3] == 1.0, "O's piece should be in ch0 when O is current player"
    assert obs[1, 0, 0] == 1.0, "X's piece should be in ch1 when O is current player"

    print("PASS test_perspective_channels")


def test_legal_moves_channel_matches_action_masks():
    """ch2 lit cells correspond exactly to True entries in action_masks()."""
    env = CNNEnv()
    env.reset()

    # Play a move to create a constrained active board
    env.game.step((0, 0, 1, 1))  # X plays centre of top-left sub-board
    env.game.current_player = -1  # force O's turn without random opponent

    obs = env._get_obs()
    masks = env.action_masks()

    for action in range(81):
        br = action // 27
        bc = (action % 27) // 9
        lr = (action % 9) // 3
        lc = action % 3
        row, col = br * 3 + lr, bc * 3 + lc
        assert obs[2, row, col] == float(masks[action]), (
            f"Mismatch at action {action} (grid {row},{col}): "
            f"ch2={obs[2, row, col]}, mask={masks[action]}"
        )
    print("PASS test_legal_moves_channel_matches_action_masks")


def test_sub_board_won_channels():
    """ch3/ch4 light up the correct 3x3 block when a sub-board is claimed."""
    env = CNNEnv()
    env.reset()

    # X wins top-left sub-board (br=0, bc=0)
    env.game.sub_board_winners[0, 0] = 1

    # From X's perspective: ch3 should light up rows 0-2, cols 0-2
    env.game.current_player = 1
    obs = env._get_obs()
    assert np.all(obs[3, 0:3, 0:3] == 1.0), "ch3 block (0:3,0:3) should be all 1 for X's win"
    assert np.all(obs[4, 0:3, 0:3] == 0.0), "ch4 block should be 0 — opponent didn't win it"

    # From O's perspective: ch4 should light up (opponent = X won it)
    env.game.current_player = -1
    obs = env._get_obs()
    assert np.all(obs[4, 0:3, 0:3] == 1.0), "ch4 block (0:3,0:3) should be all 1 from O's perspective"
    assert np.all(obs[3, 0:3, 0:3] == 0.0), "ch3 block should be 0 from O's perspective"

    print("PASS test_sub_board_won_channels")


def test_drawn_sub_board_channel():
    """ch5 lights up the correct 3x3 block for a drawn sub-board."""
    env = CNNEnv()
    env.reset()

    env.game.sub_board_winners[1, 1] = 2  # centre sub-board drawn
    obs = env._get_obs()
    assert np.all(obs[5, 3:6, 3:6] == 1.0), "ch5 block (3:6,3:6) should be all 1 for draw"
    assert obs[5, 0, 0] == 0.0, "ch5 should be 0 outside drawn sub-board"
    print("PASS test_drawn_sub_board_channel")


def test_channels_mutually_exclusive():
    """No cell can have both a piece in ch0 and ch1 simultaneously."""
    env = CNNEnv()
    env.reset()

    # Play several moves via env.step to get a realistic mid-game state
    env2 = CNNEnv()
    obs, _ = env2.reset()
    for _ in range(10):
        if env2.game.done:
            break
        moves = env2.game.get_legal_moves()
        br, bc, lr, lc = moves[0]
        obs, _, done, _, _ = env2.step(br * 27 + bc * 9 + lr * 3 + lc)

    obs = env2._get_obs()
    overlap = obs[0] * obs[1]
    assert np.all(overlap == 0), "ch0 and ch1 must not both be 1 at any cell"
    print("PASS test_channels_mutually_exclusive")


if __name__ == "__main__":
    test_obs_shape_and_dtype()
    test_initial_obs_values()
    test_board_reshape_correctness()
    test_perspective_channels()
    test_legal_moves_channel_matches_action_masks()
    test_sub_board_won_channels()
    test_drawn_sub_board_channel()
    test_channels_mutually_exclusive()
    print("\nAll tests passed.")
