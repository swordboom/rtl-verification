import random

modules = [
    "alu",
    "cache_controller",
    "memory_controller",
    "uart",
    "pcie",
    "dram_interface",
    "dma",
    "branch_predictor",
    "pipeline_controller",
    "interrupt_controller",
]

severity_levels = ["fatal", "error", "warning"]


def generate_core_features():
    m = random.choice(modules)
    s = random.choice(severity_levels)
    c = random.randint(0, 15)
    f = random.randint(1, 10)
    r = random.randint(0, 5)
    t = random.randint(1, 5)
    return m, s, c, f, r, t


if __name__ == "__main__":
    print(generate_core_features())