import numpy as np
from uttt_env import UTTTEnv

def test_reset():
    env = UTTTEnv()
    obs, info = env.reset()

    assert obs.shape == (109,), f"Expected shape (109,), got {obs.shape}"
    assert set(obs).issubset({-1, 0, 1}), f"Unexpected values in observation: {set(obs)}"
    assert info == {}, f"Expected empty info dict, got {info}"
    print("test_reset passed")


def test_step_legal():
    env = UTTTEnv()
    env.reset()

    # First move is free choice, grab any legal move and convert to flat action
    legal_moves = env.game.get_legal_moves()
    br, bc, lr, lc = legal_moves[0]
    action = br * 27 + bc * 9 + lr * 3 + lc

    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (109,), f"Expected shape (109,), got {obs.shape}"
    assert set(obs).issubset({-1, 0, 1}), f"Unexpected values in observation: {set(obs)}"
    assert not terminated, "Game should not be over after one move"
    print("test_step_legal passed")


def test_step_illegal():
    env = UTTTEnv()
    env.reset()

    # Action 0 maps to (0,0,0,0) — legal first move
    env.step(0)

    # Action 0 again — now illegal since that cell is occupied
    # and active_board has changed
    try:
        env.step(0)
        assert False, "Expected AssertionError for illegal move"
    except AssertionError:
        print("test_step_illegal passed")


if __name__ == "__main__":
    test_reset()
    test_step_legal()
    test_step_illegal()