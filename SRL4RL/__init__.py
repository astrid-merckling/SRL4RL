try:
    import SRL4RL

    SRL4RL_path = SRL4RL.__path__[0][: -len("SRL4RL")]
except ModuleNotFoundError:
    raise ModuleNotFoundError("No SRL4RL module found")


user_path = SRL4RL_path[: -len("SRL4RL/")]
