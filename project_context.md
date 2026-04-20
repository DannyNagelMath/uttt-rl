***4-13-2026
One important habit to build: Every time you come back to work on this project, your first step in PowerShell is always:
    cd C:\Users\stans\Projects\uttt-rl
    .\venv\Scripts\activate


Your environment is fully set up. Here's a quick summary of what you now have:

Python 3.11.9 installed system-wide
VS Code with the Python extension
A clean virtual environment at uttt-rl/venv that contains:

gymnasium — the RL environment framework
numpy — numerical computing
torch — PyTorch for neural networks (you'll need this later)
stable-baselines3 — pre-built RL algorithms
pygame — for rendering the game visually later

***4-20-26
We now have everything we need to write the code. Here's a precise summary of the rules we're encoding:

Board is 3×3 of 3×3 sub-boards
Players are 1 (X) and -1 (O)
First move can be anywhere
Subsequent moves must be in the sub-board corresponding to the opponent's last local cell position
If that sub-board is won or full, current player can play in any open sub-board
A sub-board is claimed when someone gets three in a row locally
A full sub-board with no winner is neutral — claimed by neither player
Game ends when someone wins the macro board (three claimed sub-boards in a row) or no sub-boards are playable (draw)


The Gymnasium contract
When you subclass Gymnasium's Env class, it expects your step() method to always return exactly five things in this order:

observation — what the agent sees (the board state)
reward — the numerical signal telling the agent how good that move was
terminated — True if the game ended naturally (win/loss/draw)
truncated — True if the episode ended artificially (e.g. hit a step limit). Almost always False for board games.
info — a dictionary for any extra debugging information. Can be empty.