# -- Import packages ---------------------------------------------------------------------------------------------------
from collections import OrderedDict
import os
import json
import numpy as np


class Diagnostics:
    """
    Diagnostics for an episode.
    """

    def __init__(self, runs):
        self.d = OrderedDict()
        self.runs = runs
        self.reset()

    def reset(self):
        """
        Reset the diagnostics dictionary.
        """
        self.d = OrderedDict()

        for r in range(self.runs):
            self.d[f'run_{r}'] = {'n_steps': [],
                                  'net_reward_to_org': [],
                                  'actionFromStart': [],
                                  'optimalPathLength': [],
                                  'optimalAction': [],
                                  'short_open_door_path': [],
                                  'long_path': [],
                                  'short_closed_door_path': [],
                                  'noGoal': [],
                                  'leaders': [],
                                  'coordinationCostsAccumulated': [],
                                  'opportunityCostsAccumulated': [],
                                  'policyInDomain_longpath': [],
                                  'policyInDomain_shortpath': [],
                                  'n_samplesModelDomain1': [],
                                  'n_samplesModelDomain2': [],
                                  'n_samplesModelDomain3': [],
                                  'n_samplesModelDomain4': [],
                                  'n_visits_per_state': np.zeros((13, 13)),
                                  'neutral_states_share': [],
                                  'shortpath_states_share': [],
                                  'longpath_states_share': [],
                                  'neutral_states_count': [],
                                  'shortpath_states_count': [],
                                  'longpath_states_count': []
                                  }

    def to_disk(self, par_name_, par_value_, start_date, start_time, runs):
        """
        Write diagnostics to disk.
        :param par_name_: parameter name
        :param par_value_: parameter value
        :param start_date: start date of the simulation run
        :param start_time: start time of the simulation run
        """
        assert type(self.d) == OrderedDict, 'ERROR: diagnostics variable must be a dictionary.'

        os.makedirs(f'outputs/traces/{start_date}/{start_time}', exist_ok=True)
        path = f'outputs/traces/{start_date}/{start_time}'
        filename = f'/dyna-q_{par_name_}_{par_value_}_{start_time}.json'

        # JSONify ndarray
        for r in range(runs):
            self.d[f'run_{r}']['n_visits_per_state'] = self.d[f'run_{r}']['n_visits_per_state'].tolist()

        with open(path + filename, "w") as f:
            json.dump(self.d, f, indent=4)

        print(f'\nSuccessfully saved to disk: {filename}\n')
