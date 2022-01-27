# -- Import packages ---------------------------------------------------------------------------------------------------
import numpy as np
import datetime as dt
import time
import sys
import os

np.set_printoptions(linewidth=200, precision=2, suppress=True)

# -- Import constant ---------------------------------------------------------------------------------------------------
from utils import DIVIDER_WIDTH


class Logger(object):
    """
    Customer logger that copies all stdout to a plain text file.
    """
    def __init__(self, start_date, start_time):
        """
        Initialize logger object
        :param start_date: start date of the simulation
        :param start_time: start time of the simulation
        """
        self.terminal = sys.stdout
        os.makedirs(f'outputs/logs/{start_date}/{start_time}', exist_ok=True)
        self.log = open(f"outputs/logs/{start_date}/{start_time}/log_{start_time}.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        pass


def print_maze(maze, dyna_params):
    """
    Print the maze environment.
    :param maze: an instance of maze
    :param dyna_params: an instance of dyna
    """

    print('\t' + '–' * DIVIDER_WIDTH)
    print('\t>> Maze parameters at start of run. '.ljust(38) +
          f'delta: {maze.DELTA}'.ljust(16) +
          f'eta: {maze.ETA}'.ljust(12) +
          f'initDoor: {maze.INIT_DOOR}'.ljust(18) +
          f'doorSwitching: {maze.DOOR_SWITCHING}'.ljust(28) +
          f'costs of coord.: {round((dyna_params.LAMBDA_ ** 2 - 1) * maze.ETA, 2)}'.ljust(30) +
          f'door status: {maze.CLOSED_DOOR_STATES}'.ljust(15))
    print('\t' + '–' * DIVIDER_WIDTH)


def cb_valid_actions(state):
    """
    Helper function to return valid actions at conditional bandit states. Used for plotting maze.
    :param state: state tuple
    :return: list of valid actions
    """

    if state == [3, 6]:
        return [2, 3]
    elif state == [9, 6]:
        return [2, 3]
    elif state == [6, 3]:
        return [0, 1]
    elif state == [6, 9]:
        return [0, 1]
    else:
        return []


def print_Q(q_values, maze, dyna, current_state, title):
    """
    Visualizes Q-values and policy on a plain text maze.
    :param q_values: Q-table
    :param maze: an instance of maze
    :param dyna: an instance of dyna
    :param current_state: current state tuple
    :param title: title string
    """

    class Bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    width = maze.MAZE_WIDTH
    height = maze.MAZE_HEIGHT
    start = maze.START_STATE
    goal = maze.GOAL_STATES
    walls = maze.WALL_STATES
    doors = maze.CLOSED_DOOR_STATES
    cb = maze.CB_STATES

    arrow_up       = '\u2191'
    arrow_down     = '\u2193'
    arrow_left     = '\u2190'
    arrow_right    = '\u2192'
    bar_vertical   = '\u2758'
    bar_horizontal = '\u2014'
    wall_symbol    = '#'
    door_symbol    = Bcolors.OKGREEN + 'X' + Bcolors.ENDC

    # Dimensions of a single cell in the maze
    cell_width = 8
    cell_height = 5

    q_values_max = np.max(q_values, axis=2)
    q_values_argmax = np.argmax(np.absolute(q_values), axis=2)
    q_values_min = np.min(q_values, axis=2)
    assert q_values_max.shape == q_values_argmax.shape == q_values_min.shape == (width, height), \
        'ERROR: Check shape of max Q-values for visualization.'

    master_string = ''

    # Assemble master string by rows in the maze, rows for each cell, and columns in the maze
    # Add 1 for row and column labels. Subtract 1 from cell height to avoid double frames.
    for row_with_labels in range(height+1):
        for row_in_cell in range(cell_height-1):
            for col_with_labels in range(width + 1):

                # Subtract 1 to index Q-table
                row_maze = max(0, row_with_labels-1)
                col_maze = max(0, col_with_labels-1)

                q_val_max = q_values_max[row_maze, col_maze]
                q_val_argmax = q_values_argmax[row_maze, col_maze]
                q_val_min = q_values_min[row_maze, col_maze]

                # If no Q-value is negative, ignore the minimum Q by setting it to infinity
                # No negative Q-values in state: take the maximum Q
                if q_val_min >= 0 and ([row_maze, col_maze] not in cb or [row_maze, col_maze] != start):
                    q_val_min = np.inf
                # No negative Q-values in state, only positives
                if q_val_min == 0 and q_val_max > 0 and [row_maze, col_maze] in cb:
                    q_val_min = np.inf
                # Only Q-values <=0 in state: take the minimum Q
                elif q_val_min < 0 and q_val_max == 0 and ([row_maze, col_maze] in cb or [row_maze, col_maze] == start):
                    pass
                # Negative and positive Q-values in state: take the maximum Q
                elif q_val_min < 0 and q_val_max > 0 and ([row_maze, col_maze] in cb or [row_maze, col_maze] == start):
                    q_val_min = np.inf

                # Adjust arrow in CB states
                if [row_maze, col_maze] in cb:
                    action_values = q_values[row_maze, col_maze, :]
                    max_action_value = max(action_values)
                    for action_index, a_value in enumerate(action_values):
                        if a_value == max_action_value and action_index in cb_valid_actions([row_maze, col_maze]):
                            q_val_argmax = action_index
                            break

                # Adjust arrow at start
                if [row_maze, col_maze] == start:
                    q_values_argmax_temp = np.argmax(q_values, axis=2)
                    q_val_argmax = q_values_argmax_temp[row_maze, col_maze]
                    # Org will choose domain with 0 over negative Q
                    if q_val_min < 0:
                        q_val_min = np.inf

                # Actual Q-value to print in this cell
                q_val_to_print = round(min(q_val_min, q_val_max), 1)

                # Flags for printing
                first_row_with_labels = row_with_labels == 0
                first_row_in_cell     = row_in_cell == 0
                first_col_with_labels = col_with_labels == 0
                middle_row_in_cell    = row_in_cell == 2

                min_Q_for_arrow = abs(q_val_to_print) >= 0.01
                if not min_Q_for_arrow:
                    q_val_argmax = None
                    q_val_to_print = 0

                # -- Rows ----------------------------------------------------------------------------------------------
                # First row
                if first_row_with_labels and first_row_in_cell and first_col_with_labels:
                    master_string += ' ' * cell_width

                # Row dividers
                if not first_row_with_labels and first_row_in_cell and first_col_with_labels:
                    master_string += ' ' * cell_width
                elif not first_row_with_labels and first_row_in_cell and not first_col_with_labels:
                    master_string += bar_horizontal * cell_width

                # Row labels
                if not first_row_with_labels and middle_row_in_cell and first_col_with_labels:
                    master_string += f'\n{row_maze: ^7}'
                elif not first_row_with_labels and not first_row_in_cell and not middle_row_in_cell and row_in_cell < cell_height - 1 and first_col_with_labels:
                    master_string += '\n' + ' ' * (cell_width - 1)

                # -- Columns -------------------------------------------------------------------------------------------
                # Column labels
                if first_row_with_labels and first_row_in_cell and col_with_labels < width:
                    master_string += f'{col_with_labels: ^8}'

                # -- Wall states ---------------------------------------------------------------------------------------
                if not first_row_with_labels and not first_row_in_cell and not first_col_with_labels and [row_maze, col_maze] in walls:
                    if col_with_labels == 1:        # first column: add vertical bar left
                        master_string += bar_vertical + wall_symbol * cell_width
                    elif col_with_labels == width:  # last column: add vertical bar right
                        master_string += bar_vertical + wall_symbol * (cell_width-2) + bar_vertical
                    else:                           # wall state in middle of maze: add vertical left
                        master_string += bar_vertical + wall_symbol * (cell_width-1)

                # -- Door states ---------------------------------------------------------------------------------------
                if not first_row_with_labels and not first_row_in_cell and [row_maze, col_maze] in doors:
                    master_string += bar_vertical + door_symbol * (cell_width-1)

                # -- Goal states --------------------------------------------------------------------------------------
                if not first_row_with_labels and not first_row_in_cell and [row_maze, col_maze] in goal:
                    if middle_row_in_cell:
                        master_string += bar_vertical + ' ' * 3 + Bcolors.OKBLUE + 'P' + Bcolors.ENDC + ' ' * 3
                    else:
                        master_string += bar_vertical + ' ' * (cell_width - 1)

                # -- Regular states ------------------------------------------------------------------------------------
                if not first_row_with_labels and not first_row_in_cell and [row_maze, col_maze] not in doors \
                        and [row_maze, col_maze] not in walls \
                        and [row_maze, col_maze] not in goal:

                    # Color the current state's Q-value RED
                    if [row_maze, col_maze] == current_state:
                        q_val_to_print_ = Bcolors.FAIL + f'{q_val_to_print:.1f}' + Bcolors.ENDC
                    else:
                        q_val_to_print_ = f'{q_val_to_print:.1f}'

                    # Best action: up
                    if row_in_cell == 1 and min_Q_for_arrow and q_val_argmax == 0:
                        master_string += bar_vertical + ' ' * 3 + arrow_up + ' ' * 3
                    elif row_in_cell == 1 and not min_Q_for_arrow and q_val_argmax == 0:
                        master_string += bar_vertical + ' ' * (cell_width - 1)
                    elif row_in_cell == 1 and (q_val_argmax != 0 or [row_maze, col_maze] in cb):
                        master_string += bar_vertical + ' ' * (cell_width - 1)
                    # Best action: down
                    elif row_in_cell == 3 and min_Q_for_arrow and q_val_argmax == 1:
                        master_string += bar_vertical + ' ' * 3 + arrow_down + ' ' * 3
                    elif row_in_cell == 3 and not min_Q_for_arrow and q_val_argmax == 1:
                        master_string += bar_vertical + ' ' * (cell_width - 1)
                    elif row_in_cell == 3 and q_val_argmax != 1:
                        master_string += bar_vertical + ' ' * (cell_width - 1)

                    # Best action not left or right
                    if middle_row_in_cell and q_val_argmax != 2 and q_val_argmax != 3:
                        if abs(q_val_to_print) >= 10:
                            master_string += bar_vertical + ' ' + q_val_to_print_ + ' ' * (1 + int(q_val_to_print >= 0))
                        else:
                            master_string += bar_vertical + ' ' * 2 + q_val_to_print_ + ' ' * (1 + int(q_val_to_print >= 0))
                    # Best action: left
                    elif middle_row_in_cell and min_Q_for_arrow and q_val_argmax == 2:
                        if abs(q_val_to_print) >= 10:
                            master_string += bar_vertical + arrow_left + ' ' + q_val_to_print_ + ' '
                        else:
                            master_string += bar_vertical + arrow_left + ' ' + q_val_to_print_ + ' ' * (1+int(q_val_to_print >= 0))
                    elif middle_row_in_cell and not min_Q_for_arrow and q_val_argmax == 2:
                        master_string += bar_vertical + ' ' * 2 + q_val_to_print_ + ' ' * (1 + int(q_val_to_print >= 0))
                    # Best action: right
                    elif middle_row_in_cell and min_Q_for_arrow and q_val_argmax == 3:
                        if abs(q_val_to_print) >= 10:
                            master_string += bar_vertical + ' ' + q_val_to_print_ + int(q_val_to_print >= 0) * ' ' + arrow_right
                        else:
                            master_string += bar_vertical + ' ' + q_val_to_print_ + ' ' + arrow_right + ' ' * int(q_val_to_print >= 0)
                    elif middle_row_in_cell and not min_Q_for_arrow and q_val_argmax == 3:
                        master_string += bar_vertical + ' ' * int(q_val_to_print >= 0) + q_val_to_print_ + ' '

        master_string += '\n'

    print(title)
    print(master_string, flush=True)


def print_start_sim(details=''):
    """
    Print a starting message for a new simulation
    :param details: string to print
    """
    print('\n' + '_' * DIVIDER_WIDTH)
    print(f'>>>> Start simulation. {details}')


def dyna_pars(dyna, with_dividers=True):
    """
    Generate a string with all dyna parameters.
    :param dyna: an instance of dyna
    :param with_dividers: whether to print divider or not (depends on verbosity level)
    :return: string with dyna parameters
    """

    if with_dividers:
        s = '\n' + '–' * DIVIDER_WIDTH + '\n'
    else:
        s = ''

    s += '\n\t>> Dyna-Q parameters. ' + \
         f'lambda: {dyna.LAMBDA_}'.ljust(15) + \
         f'tau: {dyna.TAU}'.ljust(12) + \
         f'nu: {dyna.NU}'.ljust(9) + \
         f'epsilon: {dyna.EPSILON}'.ljust(15) + \
         f'alpha: {dyna.ALPHA}'.ljust(12) + \
         f'alpha\': {dyna.ALPHA_PRIME}'.ljust(14) + \
         f'planning: {dyna.N_PLANNING}'.ljust(14) + \
         f'kappa: {dyna.KAPPA}'.ljust(14) + \
         f'phi: {dyna.PHI}'.ljust(12) + \
         f'omega: {dyna.OMEGA}'.ljust(14) + \
         f'runs: {dyna.RUNS}'.ljust(12) + \
         f'episodes: {dyna.EPISODES}'.ljust(12) + \
         '\n'

    if with_dividers:
        s += '–' * DIVIDER_WIDTH

    return s


def episode_details_exante(ep, org, maze_params, a_start, random_action, max_q, all_action_values,
                           agent_action_values):
    """
    Generate a string with all organizational details before search.
    :param ep: current org step
    :param org: an instance of organization
    :param maze_params: an instance of maze
    :param a_start: starting action
    :param random_action: indicator for random action
    :param max_q: highest Q-value
    :param all_action_values: list of all action values
    :param agent_action_values: list of an agent's action values
    :return: string with all organizational details
    """
    if max_q is not None and all_action_values is not None:
        max_q = round(max_q, 2)
        all_action_values = [round(i, 2) for i in all_action_values]
    a_start_label = '(' + maze_params.actions_labels[a_start].strip() + ')'
    s = '\n\t' + '–' * DIVIDER_WIDTH
    s += '\n\t' + '–' * DIVIDER_WIDTH
    s += '\n\t>> Organizational decision at start. Episode: ' + \
         f'\t{ep}'.ljust(10) + \
         f'Leader: {org.leader_name}'.ljust(25) + \
         f'Initial action: {a_start} {a_start_label}'.ljust(26) + \
         f'Random action: {random_action}'.ljust(25) + \
         f'\n\t\tMax action value: {max_q}'.ljust(28) + \
         f'{org.leader_name} action values: {[round(i, 2) for i in agent_action_values]}'.ljust(53) + \
         f'All action values: {all_action_values}'.ljust(58) + \
         f'Goal amt: {maze_params.GOAL_REWARD}'.ljust(30)
    if org.leader_name == 'AUTOMATION':
        s += f'\n\t>> Automation routine: {org.automation_routine}'
    s += '\n\t'
    s += '–' * DIVIDER_WIDTH

    return s


def episode_details_expost(orgStep, maze_params, orgStep_moves, orgStep_reward, coordination_costs, trace, next_state,
                           optimal_action_taken, orgStep_path, orgStep_opportunityCosts, best_potentional_net_reward):
    """
    Generate a string with all organizational details after goal reward was found.
    :param orgStep: current organizational step
    :param maze_params: an instance of maze
    :param orgStep_moves: number of moves in the organizational step
    :param orgStep_reward: accumulate reward of the organizational step
    :param coordination_costs: cumulative coorindation costs
    :param trace: trace of the state-action-reward sequence
    :param next_state: next state tuple
    :param optimal_action_taken: list of indicators whether optimal action was taken
    :param orgStep_path: string of what path the org took
    :param orgStep_opportunityCosts: opportunity costs
    :param best_potentional_net_reward: best possible net reward for this org step
    :return: string containing all organizational details
    """
    s = '\n\tOrganizational step: ' + \
        f'\t{orgStep}'.ljust(12) + \
        f'Moves: {orgStep_moves}'.ljust(16) + \
        f'Overall reward : {round(orgStep_reward, 2)}'.ljust(26) + \
        f'Goal : {round(maze_params.GOAL_REWARD, 2)}'.ljust(18) + \
        f'Coordination costs: {round(coordination_costs, 2)}'.ljust(28) + \
        f'Opportunity costs: {round(orgStep_opportunityCosts, 2)}'.ljust(26) + \
        f'Moving costs: {round(orgStep_moves * maze_params.MOVE_COST_ORG, 2)}'.ljust(22) + \
        f'Best possible: {round(best_potentional_net_reward, 2)}'.ljust(24) + \
        f'Path: {orgStep_path}'.ljust(38) + \
        f'% optimal actions: {round(np.mean(optimal_action_taken)*100, 2)}%'.ljust(28) + \
        '\n\tPath taken: {}\n'.format(' | '.join(
                [f'{sar[0]} -> {maze_params.actions_labels[sar[1]].strip()} ({round(sar[2], 2)})' for sar in
                 trace]) + ' | ' + str(next_state))

    return s


def move_details_dyna(move, move_dict_):
    """
    Generate string with all move-level details.
    NB: keep the .format() expression to avoid double quotation marks
    :param move: current move counter
    :param move_dict_: dictionary with all move-level details
    :return: string with all move-level details
    """

    s = f'\n\t\tMove: {move}'.ljust(15) + \
        'Leader: {}'.format(move_dict_['leader']).ljust(20) + \
        'A: {}'.format(move_dict_['action']).ljust(14) + \
        'Random A: {}'.format(move_dict_['Random action']).ljust(20) + \
        'Optimal A: {}'.format(move_dict_['Optimal action']).ljust(20) + \
        'Reward to agent: {}'.format(round(move_dict_['reward_to_agent'], 2)).ljust(24) + \
        '\n\t\t(ex-post)\tQ_before: {}'.format([round(i, 2) for i in move_dict_['Q_before']]).ljust(47) + \
        'Q_after: {}'.format([round(i, 2) for i in move_dict_['Q_after']]).ljust(40) + \
        'Domain: {}'.format(move_dict_['Domain']).ljust(24) + \
        'State: {}'.format(move_dict_['state']).ljust(26) + \
        'Next_state: {}'.format(move_dict_['next_state']).ljust(22) + \
        'Door status: {}'.format(move_dict_['Door status']).ljust(50) + \
        '\n\t\t\t\t\tNeutral states: {}'.format(move_dict_['neutral_states_counter']).ljust(25) + \
        'Exploration states: {}'.format(move_dict_['exploration_states_counter']).ljust(25) + \
        'Exploitation states: {}'.format(move_dict_['exploitation_states_counter'])

    return s


def run_details(run, start_time, dyna_params, org_diagnostics):
    """
    Generate string with all run-level details.
    NB: keep the .format() expression to avoid double quotation marks
    :param run: current run counter
    :param start_time: starting time of the run
    :param dyna_params: an instance of dyna
    :param org_diagnostics: dictionary with all org-level diagnostics for the run
    :return: string with all run details
    """
    s = '\t'
    s += '_' * DIVIDER_WIDTH
    s += f'\n\t>> Run {run + 1} – statistics.'.ljust(28)
    run_duration = dt.timedelta(seconds=time.time() - start_time)
    s += f'Duration: {run_duration} | ~ {run_duration * (dyna_params.RUNS - run - 1)} to go\n'

    n_orgSteps_for_average = 500
    s += '\n\tAvg. moves per org step across last {} org steps: \t{}'.format(
            n_orgSteps_for_average, round(float(np.mean(org_diagnostics['n_moves'][:n_orgSteps_for_average])), 2))
    s += '\n\tAvg. rewards per org step across last {} org steps: \t{}'.format(
            n_orgSteps_for_average, round(float(np.mean(org_diagnostics['net_reward_to_org'][:n_orgSteps_for_average])), 2))

    s += '\n\tPath to goal via:'
    s += '\n\t\tShort path (open door): {} (avg over last {} org steps in run)'.format(
            round(sum(org_diagnostics['exploration_path'][:n_orgSteps_for_average]) / n_orgSteps_for_average * 100, 2),
            n_orgSteps_for_average)
    s += '\n\t\tLong path: {} (avg over last {} org steps in run)'.format(
            round(sum(org_diagnostics['exploitation_path'][:n_orgSteps_for_average]) / n_orgSteps_for_average * 100, 2),
            n_orgSteps_for_average)
    s += '\n\t\tShort path (closed door): {} (avg over last {} org steps in run)'.format(
            round(sum(org_diagnostics['closed_door_path'][:n_orgSteps_for_average]) / n_orgSteps_for_average * 100, 2),
            n_orgSteps_for_average)

    s += '\n\tPolicy suggested:'
    s += '\n\t\tExploration path: {}%'.format(round(
            sum([1 for i in org_diagnostics['optimalPathLength'][-n_orgSteps_for_average:] if
                 i == 7]) / n_orgSteps_for_average * 100, 2))
    s += '\n\t\tExploitation path: {}%'.format(round(
            sum([1 for i in org_diagnostics['optimalPathLength'][-n_orgSteps_for_average:] if
                 i == 11]) / n_orgSteps_for_average * 100, 2))

    s += '\n\tLeaders across {} last org steps:'.format(n_orgSteps_for_average)
    s += '\n\t\t{}% Agent 1'.format(
            round(sum([1 for i in org_diagnostics['leaders'][:n_orgSteps_for_average] if 'AGENT_1' in i]) / n_orgSteps_for_average * 100, 2))
    s += '\n\t\t{}% Agent 2'.format(
            round(sum([1 for i in org_diagnostics['leaders'][:n_orgSteps_for_average] if 'AGENT_2' in i]) / n_orgSteps_for_average * 100, 2))
    s += '\n\t\t{}% Agent 3'.format(
            round(sum([1 for i in org_diagnostics['leaders'][:n_orgSteps_for_average] if 'AGENT_3' in i]) / n_orgSteps_for_average * 100, 2))
    s += '\n\t\t{}% Agent 4'.format(
            round(sum([1 for i in org_diagnostics['leaders'][:n_orgSteps_for_average] if 'AGENT_4' in i]) / n_orgSteps_for_average * 100, 2))
    s += '\n\t\t{}% Automation'.format(
            round(sum([1 for i in org_diagnostics['leaders'][:n_orgSteps_for_average] if 'AUTOMATION' in i]) / n_orgSteps_for_average * 100, 2))

    s += '\n\tAction from start:'
    s += '\n\t\t{}% UP'.format(
            round(sum([1 for i in org_diagnostics['actionFromStart'] if i == 0]) / dyna_params.EPISODES * 100, 2))
    s += '\n\t\t{}% DOWN'.format(
            round(sum([1 for i in org_diagnostics['actionFromStart'] if i == 1]) / dyna_params.EPISODES * 100, 2))
    s += '\n\t\t{}% LEFT'.format(
            round(sum([1 for i in org_diagnostics['actionFromStart'] if i == 2]) / dyna_params.EPISODES * 100, 2))
    s += '\n\t\t{}% RIGHT'.format(
            round(sum([1 for i in org_diagnostics['actionFromStart'] if i == 3]) / dyna_params.EPISODES * 100, 2))

    s += '\n\n\t'
    s += '–' * DIVIDER_WIDTH

    return s
