# -- Import packages ---------------------------------------------------------------------------------------------------
import yaml
import subprocess

# -- Import functions from other files ---------------------------------------------------------------------------------
from utils import *
from utils_IO import *
from utils_logging import *
from utils_diagnostics import *

# Simulation tag, start date, and start time for the entire simulation
TAG = ''
START_DATE = str(dt.date.today()) + TAG
START_TIME = dt.datetime.now().strftime('%H_%M_%S')


# -- Maze wrapper ------------------------------------------------------------------------------------------------------
class Maze:
    def __init__(self, startState, conditionalBanditStates, goalStates, wallStates,
                 shortpathStates, neutralStates, initDoor, doorStates, stepCost, stepsLongPath,
                 stepsShortPath, delta, eta, pi, doorSwitching):
        """
        Class for a maze, containing all the information about the environment.
        :param conditionalBanditStates: Conditional bandit states
        :param startState: Start state tuple
        :param goalStates: List of goal states
        :param wallStates: List of wall states
        :param shortpathStates: List of short path states (i.e. states on shortpath path)
        :param neutralStates: List of neutral states (i.e. neither shortpath nor longpath)
        :param initDoor: Initial state of the doors (closed, random or half)
        :param doorStates: List of door states
        :param stepCost: Cost to make a step
        :param stepsLongPath: Steps required for the risk-free path
        :param stepsShortPath: Steps required for the risky path
        :param delta: Number of times environment changes in a simulation
        :param eta: Scaling of coordination costs as a function of specialization (𝝺)
        :param pi: Scaling of reward at goal state as a function of specialization (𝝺)
        :param doorSwitching: Switching times (random or intervals)
        """

        # Ensure valid inbound values
        assert startState == [6, 6], f'ERROR: Check start state: {startState}'
        assert len(goalStates) == 4, f'ERROR: Check goal states: {goalStates}'
        assert len(doorStates) == 4, f'ERROR: Check door states : {doorStates}'
        assert delta >= 0 and isinstance(delta, int), f'ERROR: Check delta parameter value: {delta}'
        assert 0 <= eta <= 1, f'ERROR: Check eta parameter value: {eta}'
        assert doorSwitching in ['random', 'intervals'], f'ERROR: Check doorSwitching parameter value: {doorSwitching}'
        assert initDoor in ['closed', 'random', 'half'], f'ERROR: Check initDoor parameter value: {initDoor}'
        assert stepCost >= 0, f'ERROR: Check stepCost parameter value: {stepCost}'

        # Actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions_indices = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.actions_labels = ['  U  ', '  D  ', '  L  ', '  R  ']

        # Environment
        self.MAZE_WIDTH = 13
        self.MAZE_HEIGHT = 13
        self.START_STATE = startState
        self.CB_STATES = conditionalBanditStates
        self.GOAL_STATES = goalStates
        self.WALL_STATES = wallStates
        self.EXPLORATION_STATES = shortpathStates
        self.NEUTRAL_STATES = neutralStates
        self.INIT_DOOR = initDoor
        self.DOOR_STATES = deepcopy(doorStates)
        self.MOVE_COST_ORG = stepCost
        self.MOVES_EXPLOIT = stepsLongPath
        self.MOVES_EXPLORE = stepsShortPath
        self.DOOR_SWITCHING = doorSwitching
        self.MOVES_TO_DOOR = 6
        self.Q_SIZE = (self.MAZE_HEIGHT, self.MAZE_WIDTH, len(self.actions_indices))

        # Initial door states
        if self.INIT_DOOR == 'closed':
            self.CLOSED_DOOR_STATES = deepcopy(doorStates)
        elif self.INIT_DOOR == 'random':
            self.CLOSED_DOOR_STATES = [np.random.choice([i, 'open']) for i in doorStates]
        elif self.INIT_DOOR == 'half':
            self.CLOSED_DOOR_STATES = deepcopy(doorStates)
            open_doors = np.random.choice([0, 1, 2, 3], size=2, replace=False)
            self.CLOSED_DOOR_STATES[open_doors[0]] = 'open'
            self.CLOSED_DOOR_STATES[open_doors[1]] = 'open'
        else:
            self.CLOSED_DOOR_STATES = []

        # Parameters
        self.DELTA = delta
        self.ETA = eta
        self.PI = pi

        # Placeholder for goal amount
        self.GOAL_REWARD = None

        # Logging & monitoring
        self.visitedStates = []

    def make_step(self, state, action, lambda_step=1, agent_label=None):
        """
        Take action in a state.
        :param state: current state tuple [x, y]
        :param action: selected action
        :param lambda_step: number of agents
        :param agent_label: agent label (1, 2, 3 or 4)
        :return:
            [x, y]: new state
            reward: step reward
            attempted_door: indicator of whether agent attempted a clsoed door
        """

        # Ensure valid inbound values
        assert len(state) == 2 and state not in self.WALL_STATES, 'ERROR: Incorrect state in making a step.'
        assert action in maze.actions_indices, 'ERROR: Incorrect action index in making a step.'
        assert agent_label is None or agent_label in [1, 2, 3, 4], 'ERROR: Incorrect agent label when making step.'

        # Step x and y values
        x, y = deepcopy(state)
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)

        # Did the agent attempt a closed door?
        attempted_door = False
        if [x, y] in self.CLOSED_DOOR_STATES:
            attempted_door = True

        # Agent cannot step to (1) wall state, (2) previously visited states
        if [x, y] in self.WALL_STATES or ([x, y] in self.visitedStates and state != self.START_STATE):
            x, y = state

        # Agent cannot step to another agent's domain
        if agent_label is not None and state == self.START_STATE:
            if getDomainLabel([x, y], maze) not in map_agent_to_domains(agent_label, lambda_step):
                x, y = state

        # Set reward
        if [x, y] in self.GOAL_STATES:
            reward = self.GOAL_REWARD
        else:
            reward = 0

        return [x, y], reward, attempted_door


# -- Dyna-Q ------------------------------------------------------------------------------------------------------------
class DynaParams:
    def __init__(self, gamma, nu, rho, epsilon, alpha, lambda_, tau, kappa, phi, psi, runs, episodes,
                 verbose=0, vizLearning=False):
        """
        Class for the Dyna-Q agent, containing all parameters.
        :param gamma: Discount factor for Q-learning continuation value
        :param nu: Scaling of learning rate (𝛂) as a function of specialization (𝝺)
        :param rho: Number of indirect planning steps per agent per episode
        :param epsilon: Probability of taking a random action on each step
        :param alpha: Learning rate for Q-Learning when 𝝺 = 1
        :param lambda_: Number of agents in a simulation
        :param tau: Share of episodes that are automated in a simulation
        :param kappa: Time-based weight for model learning (encourages sampling long-untried transitions)
        :param phi: Percentage of coordination costs incurred under automation (𝛕) -- carrying costs
        :param psi: Costs to transition in or out of automation as a function of lambda ( ψ * 𝝺 )
        :param runs: Number of simulation runs
        :param episodes: Number of simulation episodes for a simulation run
        :param verbose: Verbosity level for logging
        :param vizLearning: Visualize learning and back-propagation in maze
        """

        # Ensure valid inbound values
        assert 0 <= gamma < 1, f'ERROR: Check gamma parameter value in yaml: {gamma}'
        assert nu >= 0, f'ERROR: Check nu parameter value in yaml: {nu}'
        assert rho >= 2, f'ERROR: Check rho parameter value in yaml: {rho}'
        assert 0 <= epsilon <= 1, f'ERROR: Check epsilon parameter value in yaml: {epsilon}'
        assert 0 <= alpha < 1, f'ERROR: Check alpha parameter value in yaml: {alpha}'
        assert lambda_ in [1, 2, 4], f'ERROR: Check lambda parameter value in yaml: {lambda_}'
        assert 0 <= tau <= 1, f'ERROR: Check tau parameter value in yaml: {tau}'
        assert kappa <= 0.001, f'ERROR: Check kappa parameter value in yaml: {kappa}'
        assert 0 <= phi <= 1, f'ERROR: Check phi parameter value in yaml: {phi}'
        assert 0 <= psi, f'ERROR: Check psi parameter value in yaml: {psi}'

        # Set parameter values
        self.GAMMA = gamma
        self.NU = nu
        self.RHO = rho
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.LAMBDA_ = lambda_
        self.TAU = deepcopy(tau)
        self.KAPPA = kappa
        self.PHI = phi
        self.PSI = psi
        self.RUNS = runs
        self.EPISODES = episodes
        self.VERBOSE = verbose
        self.VIZLEARNING = vizLearning

        # Placeholder for calculated variable
        self.ALPHA_PRIME = None
        self.TAU_AUTOMATED_EPISODES = None


# -- Model -------------------------------------------------------------------------------------------------------------
class TimeModel:
    def __init__(self, maze_params, kappa=0, rand=np.random):
        """
        Time-based model for planning in Dyna-Q and Dyna-Q+
        :param maze_params: a maze instance containing all information about the environment (used for getting all actions)
        :param kappa: weight for elapsed time in sampling reward (timeWeight=0 -> Dyna-Q, timeWeight>0 -> Dyna-Q+)
        :param rand: an instance of np.random.RandomState for sampling
        """

        # Ensure valid inbound values
        assert kappa <= 0.001, f'ERROR: Check kappa parameter value in yaml: {kappa}'

        # Strucutre: dict of ...
        #   domain
        #   dict of ...
        #       state
        #       dict of ...
        #           state, reward, time
        self.model = dict({1: dict(),
                           2: dict(),
                           3: dict(),
                           4: dict()})

        self.RAND = rand
        self.KAPPA = kappa
        self.MAZE = maze_params

        # Track the total time
        self.time_since_first_experience = 0

    def feed(self, state, action, next_state, reward, domain_label):
        """
        Feed the model with experience.
        :param state: current state as a tuple [x, y]
        :param action: selected action
        :param next_state: next state as a tuple [x, y]
        :param reward: realized reward
        :param domain_label: domain label (i.e. 1, 2, 3, 4)
        """

        # Ensure valid inbound values
        assert len(state) == 2, f'ERROR: Check state in feeding model: {state}'
        assert action in self.MAZE.actions_indices, f'ERROR: Check action in feeding model: {action}'
        assert domain_label in [1, 2, 3, 4], f'ERROR: Check domain label in feeding model: {domain_label}'

        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time_since_first_experience += 1

        # State has never been visited before
        if tuple(state) not in self.model[domain_label].keys():
            self.model[domain_label][tuple(state)] = dict()

        # Add experience to model
        self.model[domain_label][tuple(state)][action] = [list(next_state), reward, self.time_since_first_experience]

    def sample(self, domain_label):
        """
        Randomly sample from experience model.
        :param domain_label: domain label (i.e. 1, 2, 3, 4)
        :return:
            state as a tuple [x, y]
            action
            next state as a tuple [x', y']
            reward
        """

        # Ensure valid inbound values
        assert domain_label in [1, 2, 3, 4], f'ERROR: Check domain label in sampling from model: {domain_label}'

        # Select random state
        state_index = self.RAND.choice(range(len(self.model[domain_label].keys())))
        state = list(self.model[domain_label])[state_index]

        # Select random action previously taken in that state
        action_index = self.RAND.choice(range(len(self.model[domain_label][state].keys())))
        action = list(self.model[domain_label][state])[action_index]

        # Get the next state, reward, and time of last try
        next_state, reward, time_of_last_try = self.model[domain_label][state][action]

        # Adjust reward with elapsed time since last vist (S&B p. 168)
        reward += self.KAPPA * np.sqrt(self.time_since_first_experience - time_of_last_try)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward


class MasterData:
    def __init__(self):
        self.MASTER_INDEX = 0

        # Store all data on episode level
        self.initialization_row = {'start_date': START_DATE,
                                   'start_time': START_TIME,
                                   'run_id': None,
                                   'episode_id': None,
                                   'lambda': None,
                                   'alpha': None,
                                   'alpha_prime': None,
                                   'nu': None,
                                   'epsilon': None,
                                   'rho': None,
                                   'delta': None,
                                   'eta': None,
                                   'door_switching': None,
                                   'init_door': None,
                                   'tau': None,
                                   'gamma': None,
                                   'runs': None,
                                   'episodes': None,
                                   'steps': None,
                                   'net_reward_to_org': None,
                                   'optimal_action': None,
                                   'goal_amount': None,
                                   'coordination_costs': None,
                                   'opportunity_costs': None,
                                   'leader': None,
                                   'action_from_start': None,
                                   'shortpath_states_share': None,
                                   'longpath_states_share': None,
                                   'neutral_states_share': None,
                                   'path_taken': None,
                                   }

        self.data = pd.DataFrame(columns=list(self.initialization_row.keys()), index=range(RUNS * EPISODES))

    def write_to_csv(self, batch_id):
        os.makedirs(f'outputs/results/{START_DATE}', exist_ok=True)
        self.data.to_csv(f'outputs/results/{START_DATE}/{batch_id}.csv', index_label='row_index')
        print(f'Wrote {batch_id}.csv to disk.')

    def reset(self):
        self.data = pd.DataFrame(columns=list(self.initialization_row.keys()), index=range(RUNS * EPISODES))


class Organization:
    def __init__(self, dyna_params, maze_params):
        """
        Class that contains all organization-level parameters.
        :param dyna_params: an instance of dyna, containing all agent parameters.
        :param maze_params: an instance of maze
        """

        # Instantiate agent labels and independet dyna agents
        self.AGENTS_LABELS = [f'AGENT_{i + 1}' for i in range(dyna_params.LAMBDA_)]
        self.dyna_agents = [dyna_params] * len(self.AGENTS_LABELS)

        # Instantiate Q-tables
        self.q_tables = []
        for q_table_i in range(dyna_params.LAMBDA_):
            self.q_tables.append(np.full(maze_params.Q_SIZE, 0, dtype=np.float))

        self.q_table_index = None
        self.leader_name = None
        self.routine_finished = False
        self.automation_routine = None

        # Inspection metrics
        self.found_shortcut_at = False
        self.door_just_closed = False
        self.door_just_opened = False
        self.input_value = None

        # Flags for visualizing Q-table
        self.show_again_from = 0

    def select_leader(self, dyna_params_, maze_params_):
        """
        Select the leader for the episode
        :param dyna_params_: an instance of dyna
        :param maze_params_: an instance of maze
        :return:
            Q-values at start
            maximum Q-value
            all action values
        """

        # Ensure valid inbound values
        assert len(self.q_tables) == dyna_params_.LAMBDA_, f'ERROR: {dyna_params_.LAMBDA_} q-tables should exist, ' \
                                                           f'but {len(self.q_tables)} exist.'

        # -- Adjust Q-values || Parameter: LAMBDA (specialization) -----------------------------------------------------
        if dyna_params_.LAMBDA_ == 1:
            action_values_start = deepcopy(
                    self.q_tables[0][maze_params_.START_STATE[0], maze_params_.START_STATE[1], :])
            assert action_values_start.shape == (4,), f'Check Q_start shape. Current value: {action_values_start.shape}'

        elif dyna_params_.LAMBDA_ == 2:
            action_values_start = np.array(
                    [self.q_tables[0][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 0],
                     self.q_tables[0][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 1],
                     self.q_tables[1][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 2],
                     self.q_tables[1][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 3]])
            assert action_values_start.shape == (4,), f'Check Q_start shape. Current value: {action_values_start.shape}'

        elif dyna_params_.LAMBDA_ == 4:
            action_values_start = np.array(
                    [self.q_tables[0][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 0],
                     self.q_tables[1][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 1],
                     self.q_tables[2][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 2],
                     self.q_tables[3][maze_params_.START_STATE[0], maze_params_.START_STATE[1], 3]])
            assert action_values_start.shape == (4,), f'Check Q_start shape. Current value: {action_values_start.shape}'

        else:
            raise ValueError(f'ERROR: incorrect number of agents (lambda) of: {dyna_params_.LAMBDA_}')

        # Temporary Q-table of zeros and action values from start state
        Q_start = np.zeros(maze_params_.Q_SIZE, dtype=np.float)
        Q_start[maze_params_.START_STATE[0], maze_params_.START_STATE[1], :] = deepcopy(action_values_start)

        # Maximum Q-value at start
        max_q = np.max(action_values_start)

        return Q_start, max_q, action_values_start

    def select_automation_routine(self, dyna_params_, maze_params_):
        """
        Select the automation routine, if it exists.
        :param dyna_params_: an instance of dyna
        :param maze_params_: an instance of maze
        """

        # Get the automation routine as the argmax action domain
        _, _, action_values_at_start = self.select_leader(dyna_params_, maze_params_)
        a_start_index = np.argmax(action_values_at_start)
        q_table_index_a_start = map_aStart_to_qTable(a_start_index, dyna_params_.LAMBDA_)
        assert a_start_index == np.argmax(
                self.q_tables[q_table_index_a_start][maze_params_.START_STATE[0], maze_params_.START_STATE[1], :]), f'ERROR: Check automation routine selection.'

        _, self.automation_routine = optimal_path_length_routine(self.q_tables[q_table_index_a_start], maze_params_)

        # Valid routine: sequence of actions ends in goal state
        if self.automation_routine[-1][0] in maze_params_.GOAL_STATES:
            self.leader_name = 'AUTOMATION'
            self.routine_finished = False

            # Drop last (S,A) tuple from routine because that action is unnecessary
            self.automation_routine = deepcopy(self.automation_routine[:-1])

        # Invalid routine
        else:
            if dyna_params_.VERBOSE >= 2: print(f'\n\tInvalid automation routine: {self.automation_routine}')
            self.automation_routine = None


# -- Action selection --------------------------------------------------------------------------------------------------
def action_selection(state_as, q_value_as, maze_params_as, dyna_params_as):
    """
    Epsilon-greedy action selection.
    :param state_as: current state tuple [x, y]
    :param q_value_as: Q-value table
    :param maze_params_as: a maze instance containing all information about the environment
    :param dyna_params_as: a DynaParams instance containing all information about the Dyna-Q agent
    :return:
        index of e-greedy action
        random_action
    """

    # Ensure valid inbound values
    assert len(state_as) and state_as not in maze_params_as.WALL_STATES, f'ERROR: check state in ' \
                                                                         f'action selection: {state_as}'

    # Filter valid actions: No backtracking within domain and no attempts on walls
    valid_actions = []
    for action_index in maze_params_as.actions_indices:
        next_state, _, attempted_door = maze_params_as.make_step(state_as, action_index)
        # Allowed actions: state changes, attempted closed door, current state is start state
        if state_as != next_state or attempted_door or state_as == maze_params_as.START_STATE:
            valid_actions.append(action_index)

    # epsilon-greedy action
    if np.random.uniform(low=0.0, high=1.0) <= dyna_params_as.EPSILON and len(valid_actions) > 1:
        return np.random.choice(valid_actions), True, valid_actions
    # Maximum action action (breaking ties at random)
    else:
        values = q_value_as[state_as[0], state_as[1], valid_actions]
        actions = [action for action, value in zip(valid_actions, values) if value == np.max(values)]
        if len(actions) > 1:
            return np.random.choice(actions), True, valid_actions
        else:
            assert len(actions) == 1, f'ERROR: Expected single action, are {len(actions)} actions'
            return actions[0], False, valid_actions


# -- Episode -----------------------------------------------------------------------------------------------------------
def simulate_episode(org_ep, model_ep, maze_params_ep, dyna_params_ep, episode_ep, run_ep, diagnostics_instance_ep):
    """
    Function for simulating a single episode.
    :param org_ep: an organization instance containing all information about the organization
    :param model_ep: model instance for planning
    :param maze_params_ep: a maze instance containing all information about the environment
    :param dyna_params_ep: a DynaParams instance containing all information about the Dyna-Q agent
    :param episode_ep: episode index
    :param run_ep: run index
    :param diagnostics_instance_ep: diagnostics instance
    :return
        episode_steps: number of steps in the episode
        episode_reward: cumulative rewards gained in the episode
        episode_path: agent's path in this episode
        q_value: Q-values at the end of the episode
    """

    # Ensure valid inbound values
    assert dyna_params_ep.RUNS >= run_ep >= 0, f'ERROR: Check run index in simulating episode: {run_ep}'

    # Logging variables
    state = deepcopy(maze_params_ep.START_STATE)
    n_visits_per_state = np.zeros((maze_params_ep.MAZE_WIDTH, maze_params_ep.MAZE_HEIGHT))
    n_visits_per_state[state[0], state[1]] += 1.0
    episode_steps, episode_benefits = 0, 0
    shortpath_ground_counter, longpath_ground_counter = 0, 0
    optimal_action_taken_per_step = []
    next_state, Q_table = None, None

    # -- Domain selection, automation || Parameter: RHO (planning steps) & TAU (automation) -----------------------------
    # Select leading agent based on Q-values at start
    Q_start, max_q, all_action_values = org_ep.select_leader(dyna_params_=dyna_params_ep,
                                                             maze_params_=maze_params_ep)
    a_start_index, random_action, valid_actions = action_selection(state_as=state, q_value_as=Q_start,
                                                                   maze_params_as=maze_params_ep,
                                                                   dyna_params_as=dyna_params_ep)

    # Q-table index of leading agent; agent label of leader
    org_ep.q_table_index = map_aStart_to_qTable(a_start_index, dyna_params_ep.LAMBDA_)
    org_ep.leader_name = org_ep.AGENTS_LABELS[org_ep.q_table_index]

    # Ensure that action at start matches Q-table
    assert map_aStart_to_qTable(a_start_index, dyna_params_ep.LAMBDA_) == org_ep.q_table_index, \
        f'ERROR: mismatch – a_start is {a_start_index} and q-table index is {org_ep.q_table_index}'

    # Automate episode?
    if episode_ep in dyna_params_ep.TAU_AUTOMATED_EPISODES:
        org_ep.select_automation_routine(dyna_params_=dyna_params_ep, maze_params_=maze_params_ep)
        assert org_ep.leader_name == 'AUTOMATION' or 'AGENT' in org_ep.leader_name, 'ERROR: Check leader name in domain selection step.'
    else:
        org_ep.automation_routine = []

    # Make sure there is a valid starting action, valid leader, append to trace, and log episode details
    assert a_start_index in maze_params_ep.actions_indices, f'ERROR: Check a_start in domain selection: {a_start_index}'
    assert org_ep.leader_name in org_ep.AGENTS_LABELS + ['AUTOMATION'], f'ERROR: Check leader in simulating org step: {org_ep.leader_name}'

    # Trace state, action, and rewards in the episode
    episode_trace = []

    if dyna_params_ep.VERBOSE >= 2:
        agent_action_values = org_ep.q_tables[org_ep.q_table_index][maze_params_ep.START_STATE[0], maze_params_ep.START_STATE[1], :]
        print(episode_details_exante(ep=episode_ep,
                                     org=org_ep,
                                     maze_params=maze_params_ep,
                                     a_start=a_start_index,
                                     random_action=random_action,
                                     max_q=max_q,
                                     all_action_values=all_action_values,
                                     agent_action_values=agent_action_values))
        print_maze(maze_params_ep, dyna_params_ep)

    # -- Search for the goal reward -----------------------------------------------------------------------------------
    while state not in maze_params_ep.GOAL_STATES and state not in maze_params_ep.CLOSED_DOOR_STATES:

        # Track visited states
        maze_params_ep.visitedStates.append(state)

        assert org_ep.AGENTS_LABELS[org_ep.q_table_index] == org_ep.leader_name or org_ep.leader_name == 'AUTOMATION', \
            f'ERROR: incorrect index ({org_ep.q_table_index}) for episode leader ({org_ep.leader_name})'
        assert state not in maze_params_ep.WALL_STATES, f'ERROR: state cannot be a wall state: {state}'

        # -- Action selection ------------------------------------------------------------------------------------------
        action = None
        if org_ep.leader_name == 'AUTOMATION':
            action = org_ep.automation_routine[episode_steps][1]
            valid_actions = ['automation']
        elif 'AGENT' in org_ep.leader_name:
            if episode_steps == 0:  # Action selection at start state
                action = deepcopy(a_start_index)
            else:
                action, random_action, valid_actions = action_selection(state, Q_table, maze_params_ep, dyna_params_ep)
        assert action is not None, 'ERROR: action cannot be None.'

        # -- Take action -----------------------------------------------------------------------------------------------
        if 'AGENT' in org_ep.leader_name:  # Discretionary condition
            next_state, reward, door_attempted = maze_params_ep.make_step(state, action, dyna_params_ep.LAMBDA_,
                                                                          agent_label=org_ep.q_table_index + 1)
        else:  # Automation condition
            next_state, reward, door_attempted = maze_params_ep.make_step(state, action, dyna_params_ep.LAMBDA_)

        # -- Logging & monitoring --------------------------------------------------------------------------------------
        # Did the agent find the short-cut?
        if next_state in maze_params_ep.DOOR_STATES and org_ep.found_shortcut_at is False:
            org_ep.found_shortcut_at = deepcopy(episode_ep)

        # Optimal action selected?
        if state != maze_params_ep.START_STATE:
            optimal_action_taken_per_step.append(optimal_action(state, action, maze_params_ep))
        else:
            optimal_action_taken_per_step.append(1)

        # Keep track of state, action, reward, and metrics
        episode_trace.append((state, action, reward))
        episode_benefits += reward
        n_visits_per_state[next_state[0], next_state[1]] += 1.0
        shortpath_ground_counter += int(state in maze_params_ep.EXPLORATION_STATES)
        longpath_ground_counter += int(
            state not in maze_params_ep.NEUTRAL_STATES and state not in maze_params_ep.EXPLORATION_STATES)

        # -- Record experience – feed Model ----------------------------------------------------------------------------
        current_domain = getDomainLabel(state, maze_params_ep)
        if org_ep.leader_name != 'AUTOMATION':

            # Special case: starting state
            if current_domain is None:
                current_domain = action + 1

            # Realize experience: Next state when hitting door is start state
            if next_state in maze_params_ep.CLOSED_DOOR_STATES:
                next_state_model = deepcopy(maze_params_ep.START_STATE)
            else:
                next_state_model = deepcopy(next_state)

            if next_state_model == maze_params_ep.START_STATE:
                assert reward == 0, f'ERROR: Agent reward of going back to start cannot be {reward}'

            model_ep.feed(state, action, next_state_model, reward, current_domain)
            assert tuple(state) in model_ep.model[current_domain].keys(), 'ERROR: state not in model experience.'

        # -- Visualize updates -----------------------------------------------------------------------------------------
        show_indirect_updates = False
        if dyna_params_ep.VIZLEARNING and episode_ep >= org_ep.show_again_from:
            if dyna_params_ep.LAMBDA_ > 1:
                q_values_reduced = np.maximum.reduce(org_ep.q_tables) + np.minimum.reduce(org_ep.q_tables)
            else:
                q_values_reduced = deepcopy(org_ep.q_tables[0])
            print_Q(q_values_reduced, maze_params_ep, dyna_params_ep, state,
                    title=f'\n\n\t\tStep: {episode_steps}. Episode: {episode_ep}. Direct learning, before update. '
                          f'Random A: {random_action}. Valid actions: {valid_actions}. Reward: {reward}\n'
                          f'Maximum Q-values of valid actions.\n')
            org_ep.input_value = input(
                'ENTER = continue | s = skip 1 episode | j = skip 100 episodes | v = # of visits per state '
                '| d = debug next 10 episodes | i = show indirect learning updates | '
                'x y = Q-values for [x,y] state | f = finish simulatio | \n'
                'o = situation where a door opens | c = situation where a door closes \n\n')
            if org_ep.input_value == '':
                pass
            elif org_ep.input_value == 's':  # Jump single episode
                org_ep.show_again_from = episode_ep + 1
            elif org_ep.input_value == 'j':  # Jump 100 episodes
                org_ep.show_again_from = episode_ep + 100
            elif org_ep.input_value == 'v':  # Print number of visits
                n_visits_matrix = diagnostics_instance_ep['n_visits_per_state'] + n_visits_per_state
                labels = [i for i in range(n_visits_matrix.shape[0])]
                n_visits_matrix = np.insert(n_visits_matrix, 0, labels, axis=0)  # add column labels
                n_visits_matrix = np.insert(n_visits_matrix, 0, [0] + labels, axis=1)  # add row labels
                print(n_visits_matrix)
                input('ENTER = continue to next step.')
            elif org_ep.input_value == 'd':  # Debug next steps
                org_ep.show_again_from = episode_ep + 10
                input('ENTER = continue to next step.')
            elif org_ep.input_value == 'i':
                show_indirect_updates = True
                if org_ep.leader_name == 'AUTOMATION': print('No indirect learning: automation leads.')
            elif len(org_ep.input_value) == 3 or (len(org_ep.input_value) == 4 and org_ep.input_value[2] == ' '):
                s = org_ep.input_value.split()
                print(f'\tAction values for state {s}')
                print('\tDirections: ', [i.strip() for i in maze_params_ep.actions_labels])
                print('\tAction vals: ', q_values_reduced[int(s[0]), int(s[1]), :])
                input('ENTER = continue to next step.')
            elif org_ep.input_value == 'f':
                dyna_params_ep.VIZLEARNING = False
            else:
                org_ep.show_again_from += 1

        # S' transition: next state is start when hitting a closed door
        if next_state in maze_params_ep.CLOSED_DOOR_STATES:
            Q_table_next_state, _, _ = org_ep.select_leader(dyna_params_ep, maze_params_ep)
            next_state_update = deepcopy(maze_params_ep.START_STATE)
        else:
            Q_table_next_state = deepcopy(org_ep.q_tables[org_ep.q_table_index])
            next_state_update = deepcopy(next_state)

        # Remember Q-value before update
        Q_table = deepcopy(org_ep.q_tables[org_ep.q_table_index])
        q_before = deepcopy(Q_table[state[0], state[1], :])

        # Automation does not update Q-values
        if org_ep.leader_name != 'AUTOMATION':

            assert reward in [0, maze_params_ep.GOAL_REWARD], f'ERROR: Reward cannot be {reward}'

            # -- Q-Learning updates ------------------------------------------------------------------------------------
            # -- DIRECT Q-Learning update (adaptation mode) ------------------------------------------------------------
            Q_table[state[0], state[1], action] += \
                dyna_params_ep.ALPHA_PRIME * (
                        reward +
                        dyna_params_ep.GAMMA * np.max(Q_table_next_state[next_state_update[0], next_state_update[1], :])
                        - Q_table[state[0], state[1], action]
                )

            # Update Q-table
            org_ep.q_tables[org_ep.q_table_index] = deepcopy(Q_table)

            # -- INDIRECT Q-Learning update | Parameter: NU (returns to specialization) --------------------------------
            if dyna_params_ep.RHO > 0:
                for agent_index in range(dyna_params_ep.LAMBDA_):  # Each agent learns indirectly
                    agent_label = agent_index + 1

                    assert len(org_ep.q_tables) > agent_index, 'ERROR: mis-match in agent index or label during planning.'

                    # An agent can explore multiple domains at a time (if lambda is 1 or 2)
                    domains_for_agent_to_explore = map_agent_to_domains(agent_label, dyna_params_ep.LAMBDA_)
                    planning_counter = 0

                    while planning_counter < dyna_params_ep.RHO:

                        # Select a domain to sample from at random
                        domain_to_explore = np.random.choice(domains_for_agent_to_explore)
                        planning_counter += 1

                        # Check if the model contains samples in that domain
                        if model_contains_domainExperience(model_ep, domain_to_explore) and episode_ep > 0:

                            state_, action_, next_state_, reward_ = model_ep.sample(domain_to_explore)
                            assert getDomainLabel(state_, maze_params_ep) == domain_to_explore \
                                   or state_ == maze_params_ep.START_STATE, 'ERROR: Domain mismatch.'
                            assert reward_ in [0, maze_params_ep.GOAL_REWARD], f'ERROR: predicted reward cannot be {reward_}'

                            # Next state is at start: combine all Q-tables for appropriate action values at start
                            if next_state_ == maze_params_ep.START_STATE:
                                Q_table_next_state, _, _ = org_ep.select_leader(dyna_params_ep, maze_params_ep)
                            else:
                                Q_table_next_state = deepcopy(org_ep.q_tables[agent_index])

                            q_value_agent = deepcopy(org_ep.q_tables[agent_index])

                            assert reward_ in [0, maze_params_ep.GOAL_REWARD], f'ERROR: Reward cannot be {reward}'

                            # Q-learning update (lower alpha)
                            q_value_agent[state_[0], state_[1], action_] += \
                                dyna_params_ep.ALPHA * (
                                        reward_
                                        + dyna_params_ep.GAMMA * np.max(Q_table_next_state[next_state_[0], next_state_[1], :])
                                        - q_value_agent[state_[0], state_[1], action_]
                                )

                            org_ep.q_tables[agent_index] = deepcopy(q_value_agent)

                            if dyna_params_ep.VIZLEARNING and show_indirect_updates:
                                q_values_reduced = np.maximum.reduce(org_ep.q_tables) + np.minimum.reduce(
                                    org_ep.q_tables)
                                print_Q(q_values_reduced, maze_params_ep, dyna_params_ep, state_,
                                        title=f'\n\n\t\tIndirect learning, after update. Org step {episode_ep}. Step {episode_steps}. '
                                              f'State: {state_} Next_state: {next_state_} A: {action_} Reward: {reward_}\n'
                                              f'\t\tMaximum Q-values of valid actions.\n')
                                input('Press ENTER to continue.')

        # -- Populate trace --------------------------------------------------------------------------------------------
        q_after = deepcopy(Q_table[state[0], state[1], :])
        step_dict_for_logging = dict({
            'state': state,
            'next_state': next_state,
            'action': str(action) + ' (' + maze_params_ep.actions_labels[action].strip() + ')',
            'start_action': str(a_start_index) + ' (' + maze_params_ep.actions_labels[a_start_index].strip() + ')',
            'reward_to_agent': reward,
            'leader': org_ep.leader_name,
            'Q_before': q_before,
            'Q_after': q_after,
            'Random action': random_action,
            'Domain': getDomainLabel(state, maze_params_ep),
            'Door status': maze_params_ep.CLOSED_DOOR_STATES,
            'Optimal action': bool(optimal_action_taken_per_step[-1]),
            'neutral_states_counter': episode_steps - shortpath_ground_counter - longpath_ground_counter,
            'shortpath_states_counter': shortpath_ground_counter,
            'longpath_states_counter': longpath_ground_counter
        })

        if dyna_params_ep.VERBOSE >= 3: print(step_details_dyna(step=episode_steps, step_dict_=step_dict_for_logging))

        # -- Update state, and step counter ----------------------------------------------------------------------------
        state = deepcopy(next_state)
        episode_steps += 1

        # -- End episode | Parameter: TAU (automation) -----------------------------------------------------
        if org_ep.leader_name == 'AUTOMATION' and len(org_ep.automation_routine) == episode_steps:  # end of routine
            org_ep.routine_finished = True
            break

    # Track path taken
    episode_path = evaluate_path(episode_trace, next_state, maze_params_ep)

    # Reset visited states
    maze_params_ep.visitedStates = []

    # -- Organizational accounting -------------------------------------------------------------------------------------
    # Coordination costs | Parameter: ETA (coordination)
    automation_condition = org_ep.leader_name == 'AUTOMATION'
    if not automation_condition:
        episode_coordinationCosts = 1 *                      maze_params_ep.ETA * (dyna_params_ep.LAMBDA_ ** (5 / 3) - 1)
    else:
        episode_coordinationCosts = dyna_params_ep.PHI *    (maze_params_ep.ETA * (dyna_params_ep.LAMBDA_ ** (5 / 3) - 1))
    assert (episode_coordinationCosts > 0 and dyna_params_ep.LAMBDA_ > 1) \
           or (episode_coordinationCosts == 0 and dyna_params_ep.LAMBDA_ == 1), f'ERROR: Coordination costs with lambda {dyna_params_ep.LAMBDA_} and Automation {automation_condition} cannot equal {episode_coordinationCosts}'

    # Final organization net reward
    sum_of_step_costs = episode_steps * maze_params_ep.MOVE_COST_ORG
    assert sum_of_step_costs in [6, 7, 11], f'ERROR: Sum of step costs cannot equal {sum_of_step_costs}'

    # Transition costs
    if episode_ep > 0 and dyna_params_ep.PSI > 0:

        # Enter automation
        if org_ep.leader_name == 'AUTOMATION' and diagnostics_instance_ep['leaders'][-1] != 'AUTOMATION':
            transition_costs = dyna_params_ep.PSI * dyna_params_ep.LAMBDA_

        # Exit automation
        elif org_ep.leader_name != 'AUTOMATION' and diagnostics_instance_ep['leaders'][-1] == 'AUTOMATION':
            transition_costs = dyna_params_ep.PSI * dyna_params_ep.LAMBDA_

        else:
            transition_costs = 0
    else:
        transition_costs = 0

    # Add transition costs to coordination costs
    episode_coordinationCosts = episode_coordinationCosts + transition_costs

    episode_benefits = episode_benefits - sum_of_step_costs
    episode_net_reward = episode_benefits - episode_coordinationCosts

    # Worst case: no goal + no automation
    worst_case_net_reward = 0 - maze_params_ep.MOVES_TO_DOOR * maze_params_ep.MOVE_COST_ORG - episode_coordinationCosts
    assert episode_net_reward >= worst_case_net_reward, f'ERROR: Organizational net reward must be greater than 0, not {worst_case_net_reward}'

    # Best potential net reward (pi*)
    at_least_1_door_open = 'open' in maze_params_ep.CLOSED_DOOR_STATES
    if at_least_1_door_open:
        best_potentional_net_reward = maze_params_ep.GOAL_REWARD - maze_params_ep.MOVES_EXPLORE * maze_params_ep.MOVE_COST_ORG
    else:
        best_potentional_net_reward = maze_params_ep.GOAL_REWARD - maze_params_ep.MOVES_EXPLOIT * maze_params_ep.MOVE_COST_ORG
    assert best_potentional_net_reward < maze_params_ep.GOAL_REWARD, f'ERROR: best potential net reward before cannot be {best_potentional_net_reward}'

    # Track lost opportunity cost
    total_lost_opportunity = best_potentional_net_reward - episode_net_reward
    episode_opportunityCosts = total_lost_opportunity - episode_coordinationCosts
    assert total_lost_opportunity >= 0, f'ERROR: Total losses should be positive, not {total_lost_opportunity}'
    assert episode_opportunityCosts >= -0.0001, f'ERROR: Opportunity costs should be positive or zero, not {episode_opportunityCosts}'

    # Print episode level details
    if dyna_params_ep.VERBOSE >= 2: print(episode_details_expost(episode_ep, maze_params_ep, episode_steps,
                                                                 episode_net_reward, episode_coordinationCosts,
                                                                 episode_trace,
                                                                 next_state, optimal_action_taken_per_step,
                                                                 episode_path,
                                                                 episode_opportunityCosts, best_potentional_net_reward))
    assert Q_table is not None and next_state is not None, f'ERROR: Q-value is {Q_table} and next_state is {next_state}'
    assert shortpath_ground_counter + longpath_ground_counter < episode_steps, f'ERROR: explor. exploit. ground counters are too large.'

    return episode_steps, episode_net_reward, Q_table, episode_path, a_start_index, \
           np.mean(optimal_action_taken_per_step), episode_coordinationCosts, episode_opportunityCosts, \
           n_visits_per_state, shortpath_ground_counter, longpath_ground_counter


# -- Organization ------------------------------------------------------------------------------------------------------
def simulate_org(maze_params_o, dyna_params_o, diagnostics_instance_o, run_o, run_start_time_o):
    """
    Function to simulate the organization, observing and tracking the outcomes per episode.
    :param maze_params_o: a maze instance containing all information about the environment
    :param dyna_params_o: a DynaParams instance containing all information about the Dyna-Q agent
    :param diagnostics_instance_o: dictionary to track org step meta data
    :param run_o: index of the current run
    :param run_start_time_o: start time of the entire run
    """

    # Ensure valid inbound values
    assert dyna_params_o.RUNS > run_o >= 0, f'ERROR: Check run index in simulating org step: {run_o}'

    # -- Instantiate organization and model ----------------------------------------------------------------------------
    org = Organization(dyna_params_o, maze_params_o)
    model = TimeModel(maze_params=maze_params_o, kappa=dyna_params_o.KAPPA)

    # -- Execute episodes ----------------------------------------------------------------------------------
    for episode_id in range(dyna_params_o.EPISODES):

        # -- Track maze configuration ----------------------------------------------------------------------------------
        diagnostics_instance_o['maze'] = maze_params_o.__dict__
        diagnostics_instance_o['dyna'] = dyna_params_o.__dict__

        # -- Switch door state at intervals or random episodes -----------------------------------------
        # Interval switching (e.g. [500, 1000, 1500])
        if maze_params_o.DOOR_SWITCHING == 'intervals':
            DOOR_SWITCH_POINTS = switch_points_intervals(dyna_params_o.EPISODES, maze_params_o.DELTA)
        # Random switching (e.g. [236, 1298, 1782])
        elif maze_params_o.DOOR_SWITCHING == 'random':
            DOOR_SWITCH_POINTS = switch_points_random(dyna_params_o.EPISODES, maze_params_o.DELTA)
        else:
            raise ValueError(f'ERROR: check door switching mode, cannot be {maze_params_o.DOOR_SWITCHING}')
        assert len(DOOR_SWITCH_POINTS) == maze_params_o.DELTA, f'ERROR: Check switch times: {DOOR_SWITCH_POINTS}'

        # --------------------------------------------------------------------------------------------------------------
        # -- Run a single org step -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        episode_steps, episode_net_reward, q_values_afterEpisode, path_taken, a_start, opt_action_mean, coordination_costs, opportunityCosts, n_visits_per_state, shortpath_states_counter, longpath_states_counter = \
            simulate_episode(org, model, maze_params_o, dyna_params_o, episode_id, run_o, diagnostics_instance_o)

        # -- Track outcomes of the episode -----------------------------------------------------------------
        diagnostics_instance_o['net_reward_to_org'].append(episode_net_reward)
        diagnostics_instance_o['n_steps'].append(episode_steps)
        diagnostics_instance_o['leaders'].append(org.leader_name)
        diagnostics_instance_o['coordinationCostsAccumulated'].append(coordination_costs)
        diagnostics_instance_o['opportunityCostsAccumulated'].append(opportunityCosts)
        diagnostics_instance_o['long_path'].append(int('long path' in path_taken))
        diagnostics_instance_o['short_open_door_path'].append(int('short path (open door)' in path_taken))
        diagnostics_instance_o['short_closed_door_path'].append(int('short path (closed door)' in path_taken))
        diagnostics_instance_o['optimalAction'].append(opt_action_mean)
        diagnostics_instance_o['actionFromStart'].append(int(a_start))
        diagnostics_instance_o['n_samplesModelDomain1'].append(len(model.model[1]))
        diagnostics_instance_o['n_samplesModelDomain2'].append(len(model.model[2]))
        diagnostics_instance_o['n_samplesModelDomain3'].append(len(model.model[3]))
        diagnostics_instance_o['n_samplesModelDomain4'].append(len(model.model[4]))
        diagnostics_instance_o['n_visits_per_state'] += n_visits_per_state
        diagnostics_instance_o['neutral_states_share'].append((episode_steps - shortpath_states_counter - longpath_states_counter) / episode_steps)
        diagnostics_instance_o['shortpath_states_share'].append(shortpath_states_counter / episode_steps)
        diagnostics_instance_o['longpath_states_share'].append(longpath_states_counter / episode_steps)
        diagnostics_instance_o['neutral_states_count'].append(episode_steps - shortpath_states_counter - longpath_states_counter)
        diagnostics_instance_o['shortpath_states_count'].append(shortpath_states_counter)
        diagnostics_instance_o['longpath_states_count'].append(longpath_states_counter)

        # -- Master dictionary for Google BQ ---------------------------------------------------------------------------
        empty_row = deepcopy(master_data.initialization_row)
        empty_row['run_id'] = run_o
        empty_row['episode_id'] = episode_id
        empty_row['lambda'] = dyna_params_o.LAMBDA_
        empty_row['alpha'] = dyna_params_o.ALPHA
        empty_row['alpha_prime'] = dyna_params_o.ALPHA_PRIME
        empty_row['nu'] = dyna_params_o.NU
        empty_row['epsilon'] = dyna_params_o.EPSILON
        empty_row['rho'] = dyna_params_o.RHO
        empty_row['delta'] = maze_params_o.DELTA
        empty_row['eta'] = maze_params_o.ETA
        empty_row['door_switching'] = maze_params_o.DOOR_SWITCHING
        empty_row['init_door'] = maze_params_o.INIT_DOOR
        empty_row['tau'] = dyna_params_o.TAU
        empty_row['gamma'] = dyna_params_o.GAMMA
        empty_row['runs'] = RUNS
        empty_row['episodes'] = EPISODES
        empty_row['steps'] = episode_steps
        empty_row['net_reward_to_org'] = episode_net_reward
        empty_row['optimal_action'] = opt_action_mean
        empty_row['goal_amount'] = maze_params_o.GOAL_REWARD
        empty_row['coordination_costs'] = coordination_costs
        empty_row['opportunity_costs'] = opportunityCosts
        empty_row['leader'] = org.leader_name
        empty_row['action_from_start'] = a_start
        empty_row['shortpath_states_share'] = shortpath_states_counter / episode_steps
        empty_row['longpath_states_share'] = longpath_states_counter / episode_steps
        empty_row['neutral_states_share'] = (episode_steps - shortpath_states_counter - longpath_states_counter) / episode_steps
        empty_row['path_taken'] = path_taken

        # -- Update environment doors || Parameter: DELTA (stability) --------------------------------------------------
        if episode_id+1 in DOOR_SWITCH_POINTS:

            # Select two domains to change door status in
            domain_indices = np.random.choice([0, 1, 2, 3], size=2, replace=False)
            assert len(domain_indices) == 2, f'ERROR: domain indices list must be of length 2, not {len(domain_indices)}'

            # Close all doors
            maze_params_o.CLOSED_DOOR_STATES = deepcopy(maze_params_o.DOOR_STATES)

            # Open two random doors
            maze_params_o.CLOSED_DOOR_STATES[domain_indices[0]] = 'open'
            maze_params_o.CLOSED_DOOR_STATES[domain_indices[1]] = 'open'

        assert maze_params_o.CLOSED_DOOR_STATES.count('open') == 2, 'ERROR: There must always be two doors open.'

        # -- Track various org step outcomes ---------------------------------------------------------------------------
        # Optimal path length
        optimal_path_length = optimal_path_length_routine(org.q_tables[org.q_table_index], maze_params_o)[0]
        diagnostics_instance_o['optimalPathLength'].append(optimal_path_length)

        # Is the current policy in that domain an shortpath or longpath policy?
        diagnostics_instance_o['policyInDomain_longpath'].append(int(optimal_path_length == maze_params_o.MOVES_EXPLOIT))
        diagnostics_instance_o['policyInDomain_shortpath'].append(int(optimal_path_length == maze_params_o.MOVES_EXPLORE))

        # Add to master data
        assert None not in empty_row.values(), 'ERROR: None cannot be in data_dict'
        master_data.data.loc[master_data.MASTER_INDEX] = deepcopy(empty_row)
        master_data.MASTER_INDEX += 1

    # -- Log run-level details -----------------------------------------------------------------------------------------
    if dyna_params_o.VERBOSE >= 1: print(run_details(run_o, run_start_time_o, dyna_params_o, diagnostics_instance_o))


# ----------------------------------------------------------------------------------------------------------------------
# -- Running the simulation
# ----------------------------------------------------------------------------------------------------------------------

def run_simulation(maze_params, dyna_params, par_name, par_value, parX_name, parX_value, parXX_name, parXX_value):
    print_start_sim('Starting master simulation.')
    diagnostics_organization = Diagnostics(runs=RUNS)

    simulations_counter = 0
    for run in range(dyna_params.RUNS):

        # Set Dyna & maze parameters
        dyna_params, maze_params = set_params(dyna_params, maze_params, episodes=EPISODES,
                                              par_name=par_name, par_value=par_value,
                                              parX_name=parX_name, parX_value=parX_value,
                                              parXX_name=parXX_name, parXX_value=parXX_value)
        print_maze(maze_params, dyna_params)

        run_start_time = time.time()

        simulations_counter += 1
        diagnostics_organization.d[f'run_{run}']['dyna'] = dyna_params.__dict__
        print('_' * DIVIDER_WIDTH)
        print(f'>>> Simulation {simulations_counter}/{dyna_params.RUNS}')
        print(dyna_pars(dyna_params, with_dividers=False))

        # Simulate the organizational unit
        simulate_org(maze_params_o=maze_params,
                     dyna_params_o=dyna_params,
                     diagnostics_instance_o=diagnostics_organization.d[f'run_{run}'],
                     run_o=run,
                     run_start_time_o=run_start_time)

        # Reset diagnostics
        diagnostics_organization.reset()


# -- Run Dyna-Q --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    config = yaml.load(open("../config.yaml", 'r'), Loader=yaml.FullLoader)
    RUNS = config['dyna']['runs']
    EPISODES = config['dyna']['episodes']
    LOG = config['simulation']['log']
    SAVE_TO_BUCKET = config['simulation']['bucket']

    if LOG: sys.stdout = Logger(START_DATE, START_TIME)

    # set up maze and Dyna agent
    maze = Maze(startState=config['maze']['startState'],
                conditionalBanditStates=config['maze']['conditionalBanditStates'],
                goalStates=config['maze']['goalStates'],
                wallStates=config['maze']['wallStates'],
                shortpathStates=config['maze']['shortpathStates'],
                neutralStates=config['maze']['neutralStates'],
                initDoor=config['maze']['initDoor'],
                doorStates=config['maze']['doorStates'],
                stepCost=config['maze']['stepCost'],
                stepsLongPath=config['maze']['stepsLongPath'],
                stepsShortPath=config['maze']['stepsShortPath'],
                delta=config['maze']['delta'],
                eta=config['maze']['eta'],
                pi=config['maze']['pi'],
                doorSwitching=config['maze']['doorSwitching'])
    dyna = DynaParams(gamma=config['dyna']['gamma'],
                      nu=config['dyna']['nu'],
                      rho=config['dyna']['rho'],
                      epsilon=config['dyna']['epsilon'],
                      alpha=config['dyna']['alpha'],
                      lambda_=config['dyna']['lambda'],
                      tau=config['dyna']['tau'],
                      kappa=config['dyna']['kappa'],
                      phi=config['dyna']['phi'],
                      psi=config['dyna']['psi'],
                      runs=config['dyna']['runs'],
                      episodes=config['dyna']['episodes'],
                      verbose=config['simulation']['verbose'],
                      vizLearning=config['simulation']['vizLearning'])

    master_data = MasterData()

    # Parse command line arguments
    par_name_input_1, par_value_input_1, par_name_input_2, par_value_input_2, par_name_input_3, par_value_input_3 = \
        parse_commandline_args(sys.argv)

    print(f'Starting simulation with {par_name_input_1}={par_value_input_1} {par_name_input_2}={par_value_input_2} {par_name_input_3}={par_value_input_3}')
    run_simulation(maze_params=maze,
                   dyna_params=dyna,
                   par_name=par_name_input_1,
                   par_value=par_value_input_1,
                   parX_name=par_name_input_2,
                   parX_value=par_value_input_2,
                   parXX_name=par_name_input_3,
                   parXX_value=par_value_input_3)

    batch_id = f'{par_name_input_1}_{par_value_input_1}-{par_name_input_2}_{par_value_input_2}-{par_name_input_3}_{par_value_input_3}-{START_TIME}'
    master_data.write_to_csv(batch_id)

    # Copy all outputs to bucket
    if SAVE_TO_BUCKET:

        cmd = f'gsutil -m cp -r -n outputs/results/{START_DATE} gs://simulation-output/results'
        _ = subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode('utf-8')
