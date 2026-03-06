import random
import pandas as pd
from log_generator import generate_log

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
    "interrupt_controller"
]

severity_levels = ["fatal", "error", "warning"]

def assign_priority(severity, coverage_drop):

    if severity == "fatal" and coverage_drop > 8:
        return "High"

    elif severity == "error" and coverage_drop > 4:
        return "Medium"

    else:
        return "Low"

data = []

for i in range(10000):

    module = random.choice(modules)
    severity = random.choice(severity_levels)

    coverage_drop = random.randint(0,15)
    frequency = random.randint(1,10)
    recurrence = random.randint(0,5)
    fix_time = random.randint(1,5)

    log_message = generate_log(module)

    priority = assign_priority(severity, coverage_drop)

    data.append([
        i,
        module,
        severity,
        coverage_drop,
        frequency,
        recurrence,
        fix_time,
        log_message,
        priority
    ])

df = pd.DataFrame(data, columns=[
    "failure_id",
    "module",
    "severity",
    "coverage_drop",
    "frequency",
    "recurrence",
    "fix_time",
    "log_message",
    "priority"
])

df.to_csv("rtl_regression_dataset.csv", index=False)

print("Dataset generated successfully")