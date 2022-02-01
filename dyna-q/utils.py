# -- Import packages ---------------------------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy

DIVIDER_WIDTH = 300


def optimal_path_length_routine(q_values, maze):
    """
    Get the length of current optimal path given Q-values.
    :param q_values: Q-values
    :param maze: maze parameters
    :return: number of steps of the path, routine ((state, action) pairs)
    """

    max_steps_optimal_path = 100
    state = maze.START_STATE
    routine_sequence = []
    n_steps = 0

    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        if state == maze.START_STATE:
            routine_sequence.append((state, action))
            state, _, _ = maze.make_step(state, action)

            # Argmax action in new state
            action = np.argmax(q_values[state[0], state[1], :])
            routine_sequence.append((state, action))
        else:
            state, _, _ = maze.make_step(state, action)

            # Argmax action in new state
            action = np.argmax(q_values[state[0], state[1], :])
            routine_sequence.append((state, action))
        n_steps += 1
        if n_steps == max_steps_optimal_path:
            return np.nan, routine_sequence

    assert n_steps >= maze.MOVES_EXPLORE, f'ERROR: number of steps must be at least {maze.MOVES_EXPLORE}'
    assert len(routine_sequence) >= maze.MOVES_EXPLORE, f'ERROR: routine sequence must be at least {maze.MOVES_EXPLORE}'

    return n_steps, routine_sequence


def getDomainLabel(state, maze):
    """
    Return the domain LABEL (not index).
    :param state: state tuple
    :param maze: maze parameters
    :return: the domain LABEL
    """

    assert len(state) == 2 and state not in maze.WALL_STATES, 'ERROR: Cannot get domain for a wall state.'

    if state in maze.WALL_STATES or state == maze.START_STATE:
        domain = None
    elif state[0] <= 3 or state in [[4, 6], [5, 6]]:  # domain 1
        domain = 1
    elif state[0] >= 9 or state in [[7, 6], [8, 6]]:  # domain 2
        domain = 2
    elif state[1] <= 3 or state in [[6, 4], [6, 5]]:  # domain 3
        domain = 3
    elif state[1] >= 9 or state in [[6, 7], [6, 8]]:  # domain 4
        domain = 4
    else:
        raise ValueError(f'Cannot find domain for state {state}')

    return domain


def model_contains_domainExperience(model, domain):
    """
    Check whether the experience model contains samples in a particular domain.
    :param model: experience model
    :param domain: domain label
    :return: boolean indicator
    """
    samples_in_domain = len(model.model[domain].keys())
    return bool(samples_in_domain >= 1)


def switch_points_intervals(episodes, splits):
    """
    Find splitting points in a range for a given number of splits
    :param episodes: the range
    :param splits: the number of desired splits
    :return: list of split indices
    """
    indices = np.linspace(0, episodes, splits+2)
    indices = indices[1:len(indices)-1]   # trim start and end points
    indices = np.rint(indices)            # round to nearest integers
    assert len(indices) == splits, f'ERROR: More splits made ({len(indices)}) than expected: {splits}.'
    return indices


def switch_points_random(episodes, splits):
    """
    Find random splitting points in a range for a given number of splits
    :param episodes: the range
    :param splits: the number of desired splits
    :return: list of split indices
    """
    indices = np.random.choice(a=range(episodes), size=splits, replace=False)
    assert len(indices) == splits, f'ERROR: More splits made ({len(indices)}) than expected: {splits}.'
    return indices


def evaluate_path(trace, next_state, maze_params):
    """
    Evaluates path.
    :param trace: trace instance
    :param next_state: next state
    :param maze_params: maze parameters
    :return: outcome of the path
    """

    # Agent found goal
    if next_state in maze_params.GOAL_STATES:
        # Walk backwards in the maze
        for state, action, reward in reversed(trace):
            if state in maze_params.EXPLORATION_STATES:
                if len(trace) == maze_params.MOVES_EXPLORE:
                    return 'short path (open door)'
                else:
                    return 'NA'
            else:
                if len(trace) == maze_params.MOVES_EXPLOIT:
                    return 'long path'
                else:
                    return 'NA'
    elif next_state in maze_params.CLOSED_DOOR_STATES or next_state == maze_params.START_STATE:
        return 'short path (closed door)'
    else:
        ValueError('ERROR: CHECK PATH EVALUATION FUNCTION')


def optimal_action(s, action_taken, maze_params_helper):
    """
    Determine whether an action is the optimal action.
    :param s: a state
    :param action_taken: action taken
    :param maze_params_helper: an instance of maze
    :return: indicator that equals 1 if action taken was the optimal action, 0 otherwise
    """

    assert action_taken in maze_params_helper.actions_indices, 'ERROR: Action taken must be a valid action.'

    optimal_actions = dict({1: {'open_door': {(5, 6): 0,
                                              (4, 6): 0,
                                              (3, 6): 2,
                                              (3, 5): 2,
                                              (3, 4): 0,
                                              (2, 4): 0,

                                              (3, 7): 3,  # non-optimal path
                                              (3, 8): 0,
                                              (2, 8): 0,
                                              (1, 8): 2,
                                              (1, 7): 2,
                                              (1, 6): 2,
                                              (1, 5): 2},

                                'closed_door': {(5, 6): 0,
                                                (4, 6): 0,
                                                (3, 6): 3,
                                                (3, 7): 3,
                                                (3, 8): 0,
                                                (2, 8): 0,
                                                (1, 8): 2,
                                                (1, 7): 2,
                                                (1, 6): 2,
                                                (1, 5): 2,

                                                (3, 5): 3,  # non-optimal path
                                                (3, 4): 3}
                                },
                            2: {'open_door': {(7, 6): 1,
                                              (8, 6): 1,
                                              (9, 6): 3,
                                              (9, 7): 3,
                                              (9, 8): 1,
                                              (10, 8): 1,

                                              (9, 5): 2,
                                              (9, 4): 1,
                                              (10, 4): 1,
                                              (11, 4): 3,
                                              (11, 5): 3,
                                              (11, 6): 3,
                                              (11, 7): 3},
                                'closed_door': {(7, 6): 1,
                                                (8, 6): 1,
                                                (9, 6): 2,
                                                (9, 5): 2,
                                                (9, 4): 1,
                                                (10, 4): 1,
                                                (11, 4): 3,
                                                (11, 5): 3,
                                                (11, 6): 3,
                                                (11, 7): 3,

                                                (9, 7): 2,
                                                (9, 8): 2}
                                },
                            3: {'open_door': {(6, 5): 2,
                                              (6, 4): 2,
                                              (6, 3): 1,
                                              (7, 3): 1,
                                              (8, 3): 2,
                                              (8, 2): 2,

                                              (5, 3): 0,
                                              (4, 3): 2,
                                              (4, 2): 2,
                                              (4, 1): 1,
                                              (5, 1): 1,
                                              (6, 1): 1,
                                              (7, 1): 1},

                                'closed_door': {(6, 5): 2,
                                                (6, 4): 2,
                                                (6, 3): 0,
                                                (5, 3): 0,
                                                (4, 3): 2,
                                                (4, 2): 2,
                                                (4, 1): 1,
                                                (5, 1): 1,
                                                (6, 1): 1,
                                                (7, 1): 1,

                                                (7, 3): 0,
                                                (8, 3): 0}
                                },
                            4: {'open_door': {(6, 7): 3,
                                              (6, 8): 3,
                                              (6, 9): 0,
                                              (5, 9): 0,
                                              (4, 9): 3,
                                              (4, 10): 3,

                                              (7, 9): 1,
                                              (8, 9): 3,
                                              (8, 10): 3,
                                              (8, 11): 0,
                                              (7, 11): 0,
                                              (6, 11): 0,
                                              (5, 11): 0},
                                'closed_door': {(6, 7): 3,
                                                (6, 8): 3,
                                                (6, 9): 1,
                                                (7, 9): 1,
                                                (8, 9): 3,
                                                (8, 10): 3,
                                                (8, 11): 0,
                                                (7, 11): 0,
                                                (6, 11): 0,
                                                (5, 11): 0,

                                                (5, 9): 1,
                                                (4, 9): 1}
                                }
                            })

    domain = getDomainLabel(s, maze_params_helper)
    door_status = 'closed_door'
    if maze_params_helper.CLOSED_DOOR_STATES[domain-1] == 'open':
        door_status = 'open_door'

    optimal_a = str(optimal_actions[domain][door_status][tuple(s)])

    return int(str(action_taken) in optimal_a)


def set_params(dyna_params_helper, maze_params_helper, episodes, par_name, par_value, parX_name='', parX_value='',
               parXX_name='', parXX_value=''):
    """
    Set primary parameter value you want to iterate over
    :param dyna_params_helper: an instance of dyna
    :param maze_params_helper: an instance of maze
    :param episodes: number of episodes
    :param par_name: the first parameter name
    :param par_value: the first parameter value
    :param parX_name: the second parameter name (optional)
    :param parX_value: the second parameter value (optional)
    :param parXX_name: the third parameter name (optional)
    :param parXX_value: the third parameter value (optional)
    :return:
    """

    if par_name == 'delta':
        maze_params_helper.DELTA = par_value
    elif par_name == 'tau':
        dyna_params_helper.TAU = par_value
    elif par_name == 'epsilon':
        maze_params_helper.EPSILON = par_value
    elif par_name == 'lambda':  # lambda is used in both dyna and maze
        dyna_params_helper.LAMBDA_ = par_value
    else:
        raise ValueError('Unknown primary parameter name.')

    # Optional: set a second parameter
    if parX_name == '':
        pass
    elif parX_name == 'delta':
        maze_params_helper.DELTA = parX_value
    elif parX_name == 'tau':
        dyna_params_helper.TAU = parX_value
    elif parX_name == 'epsilon':
        maze_params_helper.EPSILON = parX_value
    elif parX_name == 'lambda':  # lambda is used in both dyna and maze
        dyna_params_helper.LAMBDA_ = parX_value
    else:
        raise ValueError('Unknown secondary parameter name.')

    # Optional: set a third parameter
    if parXX_name == '':
        pass
    elif parXX_name == 'delta':
        maze_params_helper.DELTA = parXX_value
    elif parXX_name == 'tau':
        dyna_params_helper.TAU = parXX_value
    elif parXX_name == 'epsilon':
        maze_params_helper.EPSILON = parXX_value
    elif parXX_name == 'lambda':  # lambda is used in both dyna and maze
        dyna_params_helper.LAMBDA_ = parXX_value
    else:
        raise ValueError('Unknown third parameter name.')

    # -- Set calculated values -----------------------------------------------------------------------------------------

    maze_params_helper.GOAL_REWARD = round(10 * np.log(dyna_params_helper.LAMBDA_ + 1), 2)  # 10 scales benefits

    # Specialization learning rate
    dyna_params_helper.ALPHA_PRIME = deepcopy(dyna_params_helper.ALPHA + dyna_params_helper.NU * np.log(dyna_params_helper.LAMBDA_))
    dyna_params_helper.ALPHA_PRIME = round(dyna_params_helper.ALPHA_PRIME, 2)
    dyna_params_helper.TAU_AUTOMATED_EPISODES = np.random.choice(range(episodes), int(dyna_params_helper.TAU * episodes), replace=False)

    # Re-initialize doors at the beginning of each run
    if maze_params_helper.INIT_DOOR == 'half':
        maze_params_helper.CLOSED_DOOR_STATES = deepcopy(maze_params_helper.DOOR_STATES)
        open_doors = np.random.choice([0, 1, 2, 3], size=2, replace=False)
        maze_params_helper.CLOSED_DOOR_STATES[open_doors[0]] = 'open'
        maze_params_helper.CLOSED_DOOR_STATES[open_doors[1]] = 'open'
    elif maze_params_helper.INIT_DOOR == 'random':
        maze_params_helper.CLOSED_DOOR_STATES = [np.random.choice([i, 'open']) for i in maze_params_helper.DOOR_STATES]

    assert 0 <= dyna_params_helper.ALPHA_PRIME <= 1, f'ERROR: Check alpha prime parameter: {dyna_params_helper.ALPHA_PRIME}'
    assert dyna_params_helper.ALPHA_PRIME >= dyna_params_helper.ALPHA, f'ERROR: Check alpha prime {dyna_params_helper.ALPHA_PRIME} and alpha {dyna_params_helper.ALPHA}'
    assert int(dyna_params_helper.TAU * episodes) - 1 <= len(dyna_params_helper.TAU_AUTOMATED_EPISODES) <= int(dyna_params_helper.TAU * episodes) + 1, f'ERROR: Check length of automated episodes: {len(dyna_params_helper.TAU_AUTOMATED_EPISODES)}'
    assert maze_params_helper.CLOSED_DOOR_STATES.count('open') == 2, 'ERROR: There must always be two doors open.'

    return deepcopy(dyna_params_helper), deepcopy(maze_params_helper)


def map_aStart_to_qTable(a_start, lambda_tmp):
    """
    Returns the qTable index for the initial action.
    :param a_start: action from start state
    :param lambda_tmp: lambda value
    :return: domain label
    """

    assert a_start in [0, 1, 2, 3], f'ERROR: Check agent label: {a_start}'
    assert lambda_tmp in [1, 2, 3, 4], f'ERROR: Check lambda in mapping function: {lambda_tmp}'

    if lambda_tmp == 1:    # no specialization
        return 0

    elif lambda_tmp == 2:  # some specialization
        m = dict({0: 0,
                  1: 0,
                  2: 1,
                  3: 1})
        return m[a_start]

    elif lambda_tmp == 4:   # full specialization
        m = dict({0: 0,
                  1: 1,
                  2: 2,
                  3: 3})
        return m[a_start]

    else:
        raise ValueError("ERROR: Could not map starting action to q-table.")


def map_agent_to_domains(agent_label, lambda_tmp):
    """
    Returns the domain(s) that an agent is allowed to explore.
    :param agent_label: agent label
    :param lambda_tmp: lambda value
    :return: domain label
    """

    assert agent_label in [1, 2, 3, 4], f'ERROR: Check agent label: {agent_label}'
    assert lambda_tmp in [1, 2, 3, 4], f'ERROR: Check lambda in mapping function: {lambda_tmp}'

    if lambda_tmp == 1:  # no specialization
        return [1, 2, 3, 4]
    elif lambda_tmp == 2:  # some specialization
        m = dict({1: [1, 2],
                  2: [3, 4]})
        return m[agent_label]
    elif lambda_tmp == 4:
        m = dict({1: [1],
                  2: [2],
                  3: [3],
                  4: [4]})
        return m[agent_label]
    else:
        raise ValueError("ERROR: Could not map agent to domain.")


def mc_perm_test(xs, ys, m):
    """
    Exact Monte Carlo permutation test (two-sided) (Ernst, 2004)
    :param xs: x array
    :param ys: y array
    :param m: number of permutations
    :return: p-value
    """
    n, bs = len(xs), 0
    d_original = np.abs(np.mean(xs) - np.mean(ys))
    combined = np.concatenate([xs, ys])
    for j in range(m):
        np.random.shuffle(combined)  # shuffle in place
        d_permute = np.abs(np.mean(combined[:n]) - np.mean(combined[n:]))
        bs += int(d_permute >= d_original)
    p_val = (bs + 1) / (m + 1)
    return round(p_val, 15)


def parse_commandline_args(args):
    """
    Parse command line arguments
    :param args: list of arguments
    :return: parameter names and values
    """

    assert len(args) == 7, f'ERROR: pass exactly 7 arguments to launch the simulation, not {len(args)}.'

    expected_par_names = ['lambda', 'tau', 'delta']

    assert args[1] in expected_par_names and args[3] in expected_par_names and args[5] in expected_par_names, \
        f'ERROR: parsed unexpected parameter name, one of {args[1], args[3], args[5]}'

    # Parameter names
    par_name_input_1 = str(args[1])
    par_name_input_2 = str(args[3])
    par_name_input_3 = str(args[5])

    # Parameter values
    try:
        par_value_input_1 = int(args[2])
    except ValueError:
        par_value_input_1 = float(args[2])
    try:
        par_value_input_2 = int(args[4])
    except ValueError:
        par_value_input_2 = float(args[4])
    try:
        par_value_input_3 = int(args[6])
    except ValueError:
        par_value_input_3 = float(args[6])

    return par_name_input_1, par_value_input_1, par_name_input_2, par_value_input_2, par_name_input_3, par_value_input_3



