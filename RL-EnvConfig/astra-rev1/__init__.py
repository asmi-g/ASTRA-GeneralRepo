from gymnasium.envs.registration import register

register(
    id="astra-rev1/GridWorld-v0",
    entry_point="astra-rev1.envs:GridWorldEnv",
)
