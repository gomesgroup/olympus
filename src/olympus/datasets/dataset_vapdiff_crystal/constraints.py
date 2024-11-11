#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
from pulp import *

from olympus import __home__

# max molarity constraints
max_mol_constr = pickle.load(open(f'{__home__}/datasets/dataset_vapdiff_crystal/max_molarity_constraints.pkl', 'rb'))

# convex hull with chemname column headers
df_hull = pd.read_csv(f'{__home__}/datasets/dataset_vapdiff_crystal/convex_hull_chemnames.csv')


def known_constraints(param_vec):

    organic = param_vec[0]
    organic_molarity = param_vec[1]
    solvent = param_vec[2]
    solvent_molarity = param_vec[3]
    inorganic_molarity = param_vec[4]
    acid_molarity = param_vec[5]
    alpha_vial_volume = param_vec[6]
    beta_vial_volume = param_vec[7]
    reaction_time = param_vec[8]
    reaction_temperature = param_vec[9]

    # check max molarity constraints first
    if organic_molarity > max_mol_constr['organic_molarity_max'][organic]:
        return False

    if inorganic_molarity > max_mol_constr['inorganic_molarity_max']['Lead Diiodide']:
        return False

    if solvent_molarity > max_mol_constr['solvent_molarity_max'][solvent]:
        return False

    if acid_molarity > max_mol_constr['acid_molarity_max']['Formic Acid']:
        return False

    # check convex hull
    target_molarities = {
        organic:organic_molarity, 'Lead Diiodide':inorganic_molarity, solvent:solvent_molarity,
        'Formic Acid':acid_molarity,
    }
    good_check = backward_check(target_molarities, alpha_vial_volume, df_hull)
    if not good_check:
        return False
    else:
        return True


def backward_check(target_molarities, alpha_vial_volume, df_hull):
    chemnames = [str(c) for c in df_hull.columns]
    assert set(chemnames).issuperset(set(target_molarities.keys()))
    target_molarities_padded = np.zeros(len(df_hull.columns))
    
    for i, chemname in enumerate(chemnames):
        try:
            target_molarities_padded[i] = target_molarities[chemname]
        except KeyError:
            continue
    
    n_ss, n_chem = df_hull.shape
    
    # Create LP problem
    prob = LpProblem("backward_check", LpMinimize)
    
    # Create variables
    v = [LpVariable(f"v{i}", lowBound=0) for i in range(n_ss)]
    
    # Objective function (can be arbitrary since we only care about feasibility)
    prob += lpSum(v)
    
    # Add constraints
    # Volume constraint
    prob += lpSum(v) == alpha_vial_volume
    
    # Molarity constraints
    for j in range(n_chem):
        prob += lpSum(v[i] * df_hull.values[i][j] for i in range(n_ss)) == alpha_vial_volume * target_molarities_padded[j]
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=False))
    
    return prob.status == LpStatusOptimal
