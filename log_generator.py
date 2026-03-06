import random

def generate_log(module):

    logs = [
        f"Assertion failure detected in {module}",
        f"Pipeline hazard detected in {module}",
        f"Invalid memory access violation in {module}",
        f"Cache coherency violation in {module}",
        f"DMA transfer error in {module}",
        f"Branch misprediction detected in {module}",
        f"Buffer overflow detected in {module}",
        f"Protocol violation detected in {module}"
    ]

    return random.choice(logs)