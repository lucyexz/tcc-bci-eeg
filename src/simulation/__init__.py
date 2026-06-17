def run_simulation(*args, **kwargs):
    from src.simulation.pygame_simulator import run_simulation as _run
    return _run(*args, **kwargs)

__all__ = ["run_simulation"]
