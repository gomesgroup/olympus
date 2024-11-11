import os, json
import pandas as pd

raw_data = pd.read_csv("raw_data.csv")

config = {
    "constraints": {
        "parameters": "none",
        "measurements": "none",
        "known": "no"
    },
    "parameters": [],
    "measurements": [],
    "default_goal": "maximize"
}

parameter_names = [
    "base",
    "ligand",
    "aryl_halide",
    "additive"
]

objective_name = "yield"

data = raw_data[parameter_names + [objective_name]]
# Convert nan to None
data = data.fillna("None")

for parameter_name in parameter_names:
    param_data = data[parameter_name].tolist()
    param_options = list(set(param_data))
    param_descriptors = []
    
    config["parameters"].append({
        "name": parameter_name,
        "type": "categorical",
        "options": param_options,
        "descriptors": param_descriptors
    })

config["measurements"].append({
    "name": objective_name,
    "type": "continuous"
})

# save data to  data.csv
data.to_csv("data.csv", index=False)

# save config to config.json
json.dump(config, open("config.json", "w"), indent=4)