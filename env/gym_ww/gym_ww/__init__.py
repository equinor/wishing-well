from gym.envs.registration import register


register(
    id='well-plot-3-v0',
    entry_point='gym_ww.envs:WellPlot3Env'
)

register(
    id='well-plot-5-v0',
    entry_point='gym_ww.envs:WellPlot5Env',
)

register(
    id='well-plot-16-v0',
    entry_point='gym_ww.envs:WellPlot16Env',
)

register(
    id='well-plot-21-v0',
    entry_point='gym_ww.envs:WellPlot21Env',
)

