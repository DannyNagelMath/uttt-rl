from sb3_contrib import MaskablePPO
from uttt_env import UTTTEnv

env = UTTTEnv()

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

model.save("uttt_maskable_ppo")