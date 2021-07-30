
try:
    import SRL4RL
    SRL4RL_path = SRL4RL.__path__[0][:-len('SRL4RL')]
except:
    print('\n Cannot import SRL4RL!')


user_path = SRL4RL_path[:-len('SRL4RL/')]
