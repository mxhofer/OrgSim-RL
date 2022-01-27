# -- Import packages ---------------------------------------------------------------------------------------------------
from collections import OrderedDict
import os
import pandas as pd
import gzip
import pickle

# -- Import constants & functions --------------------------------------------------------------------------------------
from utils import DIVIDER_WIDTH
from utils_logging import print_maze, move_details_dyna
from dynaQ import START_DATE, START_TIME


def write_results(fname, data, dyna_params_, par_name_, par_values_, parX_name='', parX_values=None,
                  over_episode=True):
    """
    Write results to a csv file.
    :param fname: file name
    :param data: simulation data
    :param dyna_params_: an isntance of dyna
    :param par_name_: the first parameter name
    :param par_values_: the first parameter values
    :param parX_name: the second parameter name
    :param parX_values: the second parameter values
    :param over_episode: indicator of whether to write results by episode or one parameter against the other
    """

    os.makedirs(f'outputs/results/{START_DATE}/{START_TIME}', exist_ok=True)
    path = f'outputs/results/{START_DATE}/{START_TIME}/'

    if over_episode:
        data_df = pd.DataFrame(data, columns=[str(i) for i in range(dyna_params_.EPISODES)],
                               index=[f'{par_name_}_' + str(i) for i in par_values_])
        data_df.to_csv(path + fname)
    elif not over_episode and parX_name == '':
        data_df = pd.DataFrame(data, columns=['Aggregate metric'],
                               index=[f'{par_name_}_' + str(i) for i in par_values_])
        data_df.to_csv(path + fname)
    else:
        data_df = pd.DataFrame(data, columns=[f'{parX_name}_' + str(i) for i in parX_values],
                               index=[f'{par_name_}_' + str(i) for i in par_values_])
        data_df.to_csv(path + fname)
