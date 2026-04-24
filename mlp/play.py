import sys, os

if getattr(sys, 'frozen', False):
    # PyInstaller bundle: modules are pre-collected; data files live in sys._MEIPASS
    _BUNDLE_DIR = sys._MEIPASS
    sys.path.insert(0, _BUNDLE_DIR)
else:
    _DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_DIR))  # project root -> finds uttt_game
    sys.path.insert(0, _DIR)                   # mlp/ -> finds uttt_env, utils

import numpy as np
import pygame

from sb3_contrib import MaskablePPO
from uttt_game import UTTTGame
from uttt_env import UTTTEnv

# ── Layout constants ──────────────────────────────────────────────────────────
WINDOW_SIZE   = 630
MARGIN        = 20          # outer margin around the macro board
MACRO_GAP     = 6           # gap between sub-boards
SUB_GAP       = 2           # gap between cells within a sub-board

# Derived
BOARD_SIZE    = WINDOW_SIZE - 2 * MARGIN
SUB_SIZE      = (BOARD_SIZE - 2 * MACRO_GAP) // 3          # sub-board pixel size
CELL_SIZE     = (SUB_SIZE  - 2 * SUB_GAP)   // 3          # cell pixel size

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG          = (30,  30,  30)
C_GRID_MACRO  = (80,  80,  80)
C_GRID_SUB    = (55,  55,  55)
C_ACTIVE_HL   = (60, 120, 200)   # blue highlight for active sub-board(s)
C_X           = (220,  80,  80)  # red for X (agent)
C_O           = (80,  180, 120)  # green for O (human)
C_WON_X_BG   = (80,  30,  30)   # dark red tint for X-won sub-board
C_WON_O_BG   = (30,  70,  40)   # dark green tint for O-won sub-board
C_DRAWN_BG   = (50,  50,  50)
C_TEXT        = (220, 220, 220)
C_STATUS_BG   = (20,  20,  20)

STATUS_H      = 40           # height of status bar at bottom
TOTAL_H       = WINDOW_SIZE + STATUS_H

if getattr(sys, 'frozen', False):
    MODEL_PATH = os.path.join(sys._MEIPASS, "best_model.zip")
else:
    MODEL_PATH = os.path.join(os.path.dirname(_DIR), "best_models", "run2_selfplay_500000.zip")


def sub_board_origin(br, bc):
    """Top-left pixel of sub-board (br, bc)."""
    x = MARGIN + bc * (SUB_SIZE + MACRO_GAP)
    y = MARGIN + br * (SUB_SIZE + MACRO_GAP)
    return x, y


def cell_origin(br, bc, lr, lc):
    """Top-left pixel of cell (lr, lc) inside sub-board (br, bc)."""
    sx, sy = sub_board_origin(br, bc)
    x = sx + lc * (CELL_SIZE + SUB_GAP)
    y = sy + lr * (CELL_SIZE + SUB_GAP)
    return x, y


def pixel_to_cell(px, py):
    """
    Convert a pixel coordinate to (br, bc, lr, lc), or None if outside the board.
    """
    px -= MARGIN
    py -= MARGIN
    if px < 0 or py < 0 or px >= BOARD_SIZE or py >= BOARD_SIZE:
        return None

    bc_f = px / (SUB_SIZE + MACRO_GAP)
    br_f = py / (SUB_SIZE + MACRO_GAP)
    bc = int(bc_f)
    br = int(br_f)
    if br >= 3 or bc >= 3:
        return None

    # Position within the sub-board
    lx = px - bc * (SUB_SIZE + MACRO_GAP)
    ly = py - br * (SUB_SIZE + MACRO_GAP)
    if lx >= SUB_SIZE or ly >= SUB_SIZE:
        return None   # click landed in the macro gap

    lc = int(lx / (CELL_SIZE + SUB_GAP))
    lr = int(ly / (CELL_SIZE + SUB_GAP))
    if lr >= 3 or lc >= 3:
        return None

    return br, bc, lr, lc


def draw_x(surface, cx, cy, size, colour, alpha=255):
    """Draw an X symbol centred at (cx, cy)."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    pad = size * 0.15
    w = max(2, size // 8)
    pygame.draw.line(s, (*colour, alpha), (pad, pad), (size - pad, size - pad), w)
    pygame.draw.line(s, (*colour, alpha), (size - pad, pad), (pad, size - pad), w)
    surface.blit(s, (cx - size // 2, cy - size // 2))


def draw_o(surface, cx, cy, size, colour, alpha=255):
    """Draw an O symbol centred at (cx, cy)."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    w = max(2, size // 8)
    pygame.draw.circle(s, (*colour, alpha), (size // 2, size // 2),
                       int(size * 0.38), w)
    surface.blit(s, (cx - size // 2, cy - size // 2))


def draw_board(surface, game, font_small, font_status, status_msg):
    surface.fill(C_BG)

    # ── Determine which sub-boards are active ────────────────────────────────
    active_set = set()
    if not game.done:
        if game.active_board is not None:
            active_set.add(game.active_board)
        else:
            for br in range(3):
                for bc in range(3):
                    if game.sub_board_winners[br, bc] == 0:
                        active_set.add((br, bc))

    # ── Draw sub-boards ───────────────────────────────────────────────────────
    for br in range(3):
        for bc in range(3):
            sx, sy = sub_board_origin(br, bc)
            winner = game.sub_board_winners[br, bc]

            # Background tint for claimed sub-boards
            if winner == 1:
                bg = C_WON_X_BG
            elif winner == -1:
                bg = C_WON_O_BG
            elif winner == 2:
                bg = C_DRAWN_BG
            else:
                bg = C_BG

            pygame.draw.rect(surface, bg, (sx, sy, SUB_SIZE, SUB_SIZE))

            # Active highlight border
            if (br, bc) in active_set:
                pygame.draw.rect(surface, C_ACTIVE_HL,
                                 (sx - 3, sy - 3, SUB_SIZE + 6, SUB_SIZE + 6), 3)

            # ── Draw cells ────────────────────────────────────────────────────
            for lr in range(3):
                for lc in range(3):
                    cx_px, cy_px = cell_origin(br, bc, lr, lc)
                    # Cell background
                    pygame.draw.rect(surface, C_GRID_SUB,
                                     (cx_px, cy_px, CELL_SIZE, CELL_SIZE))
                    val = game.board[br, bc, lr, lc]
                    centre_x = cx_px + CELL_SIZE // 2
                    centre_y = cy_px + CELL_SIZE // 2
                    sym_size = int(CELL_SIZE * 0.7)
                    if val == 1:
                        draw_x(surface, centre_x, centre_y, sym_size, C_X)
                    elif val == -1:
                        draw_o(surface, centre_x, centre_y, sym_size, C_O)

            # ── Large overlay for won/drawn sub-boards ────────────────────────
            centre_sx = sx + SUB_SIZE // 2
            centre_sy = sy + SUB_SIZE // 2
            big = int(SUB_SIZE * 0.75)
            if winner == 1:
                draw_x(surface, centre_sx, centre_sy, big, C_X, alpha=160)
            elif winner == -1:
                draw_o(surface, centre_sx, centre_sy, big, C_O, alpha=160)
            elif winner == 2:
                txt = font_small.render("draw", True, (120, 120, 120))
                surface.blit(txt, txt.get_rect(center=(centre_sx, centre_sy)))

    # ── Macro grid lines ──────────────────────────────────────────────────────
    for i in range(1, 3):
        x = MARGIN + i * (SUB_SIZE + MACRO_GAP) - MACRO_GAP // 2
        pygame.draw.line(surface, C_GRID_MACRO,
                         (x, MARGIN), (x, MARGIN + BOARD_SIZE), 2)
        y = MARGIN + i * (SUB_SIZE + MACRO_GAP) - MACRO_GAP // 2
        pygame.draw.line(surface, C_GRID_MACRO,
                         (MARGIN, y), (MARGIN + BOARD_SIZE, y), 2)

    # ── Status bar ────────────────────────────────────────────────────────────
    pygame.draw.rect(surface, C_STATUS_BG,
                     (0, WINDOW_SIZE, WINDOW_SIZE, STATUS_H))
    status_surf = font_status.render(status_msg, True, C_TEXT)
    surface.blit(status_surf, status_surf.get_rect(
        midleft=(MARGIN, WINDOW_SIZE + STATUS_H // 2)))


def get_status(game, waiting_for_agent=False):
    if game.done:
        if game.winner == 1:
            return "Agent (X) wins!  Press any key to quit."
        elif game.winner == -1:
            return "You (O) win!  Press any key to quit."
        else:
            return "Draw!  Press any key to quit."
    if waiting_for_agent:
        return "Agent is thinking..."
    return "Your turn (O) — click a highlighted cell"


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, TOTAL_H))
    pygame.display.set_caption("Ultimate Tic-Tac-Toe  |  You: O   Agent: X")
    clock = pygame.time.Clock()

    font_small  = pygame.font.SysFont("segoeui", 14)
    font_status = pygame.font.SysFont("segoeui", 18)

    # ── Load model and set up env/game ────────────────────────────────────────
    print("Loading model...")
    model = MaskablePPO.load(MODEL_PATH)
    print("Model loaded.")

    game = UTTTGame()
    env  = UTTTEnv()
    env.game = game          # point env at the shared game instance

    # ── Agent takes the first move (X goes first) ─────────────────────────────
    waiting_for_agent = True

    running = True
    while running:
        # ── Agent move ────────────────────────────────────────────────────────
        if waiting_for_agent and not game.done:
            draw_board(screen, game, font_small, font_status,
                       get_status(game, waiting_for_agent=True))
            pygame.display.flip()

            obs   = env._get_obs()
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            br = action // 27
            bc = (action % 27) // 9
            lr = (action % 9)  // 3
            lc = action % 3
            game.step((br, bc, lr, lc))
            waiting_for_agent = False

        # ── Render ────────────────────────────────────────────────────────────
        draw_board(screen, game, font_small, font_status,
                   get_status(game, waiting_for_agent=False))
        pygame.display.flip()

        # ── Event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and game.done:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not game.done:
                if event.button == 1:   # left click
                    cell = pixel_to_cell(*event.pos)
                    if cell is not None:
                        br, bc, lr, lc = cell
                        legal = game.get_legal_moves()
                        if (br, bc, lr, lc) in legal:
                            game.step((br, bc, lr, lc))
                            if not game.done:
                                waiting_for_agent = True

        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()