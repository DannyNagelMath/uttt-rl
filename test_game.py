from uttt_game import UTTTGame

def test_full_game():
    game = UTTTGame()
    obs = game.reset()
    move_count = 0

    print("=== Simulating a full random game ===\n")

    while not game.done:
        moves = game.get_legal_moves()
        assert len(moves) > 0, "No legal moves but game not done!"

        # Pick the first legal move (not random, but deterministic and testable)
        action = moves[0]
        obs, reward, terminated, truncated, info = game.step(action)
        move_count += 1

    game.render()
    print(f"Game over after {move_count} moves.")
    print(f"Winner: {game.winner} (1=X, -1=O, 0=draw)")
    assert game.done == True
    assert game.winner in [1, -1, 0]
    print("\nAll assertions passed!")

def test_free_choice_after_completed_board():
    game = UTTTGame()
    game.reset()

    print("=== Testing free choice after completed board ===\n")

    game.board[0, 0] = np.array([[0, -1, 1],
                                  [1, -1, -1],
                                  [-1, 1, 1]])
    
    game.active_board = None
    game.current_player = 1
    game.step((0,0,0,0))
     
    moves = game.get_legal_moves()
    print(f"Active board is drawn. Number of legal moves: {len(moves)}")
    print(f"Free choice (should be 72): {len(moves) == 72}")
    assert len(moves) == 72, f"Expected 72 legal moves, got {len(moves)}"
    print("\nAll assertions passed!")

import numpy as np

test_full_game()
print("\n" + "="*40 + "\n")
test_free_choice_after_completed_board()