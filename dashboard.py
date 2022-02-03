# Import packages
import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import itertools
from scipy import stats
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_file(
    filename='.streamlit/<filename>'
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)


# Perform query.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache to hash the return value.
    rows_query = [dict(r) for r in rows_raw]
    return rows_query


# ----------------------------------------------------------------------------------------------------------------------
def format_parameters(query_result, lambdas=None, taus=None):
    """
    Format parameters from BigQuery result
    :param query_result: result from BQ
    :param lambdas: list of lambda values
    :param taus: list of tau values
    :return: parameter string
    """

    # It's the pythonic way ...
    if taus is None:
        taus = []
    if lambdas is None:
        lambdas = []

    pars_to_show = ['start_date', 'start_time', 'lambda', 'alpha', 'alpha_prime', 'nu', 'epsilon', 'rho',
                    'delta', 'eta', 'door_switching', 'init_door', 'tau', 'gamma', 'runs', 'episodes']
    taus = [round(t, 2) for t in taus]
    index_values = []
    pars_dict = {'Parameter value': []}
    for k, v in query_result[0].items():
        if k in pars_to_show:
            if k == 'lambda' and len(lambdas) > 0:
                pars_dict['Parameter value'].append(str(lambdas))
                index_values.append('lambda')
            elif k == 'tau' and len(taus) > 0:
                pars_dict['Parameter value'].append(str(taus))
                index_values.append('tau')
            elif type(v) == str or type(v) == datetime.date:
                pars_dict['Parameter value'].append(str(v))
                index_values.append(k)
            else:
                pars_dict['Parameter value'].append(str(round(v, 2)))
                index_values.append(k)

    return pd.DataFrame(pars_dict, index=index_values)


def format_validation(query_result):
    """
    Get the
    :param query_result:
    :return:
    """

    observed_n_steps = []
    for k, v in query_result[0].items():
        observed_n_steps.append(v)

    observed_n_steps = list(set(observed_n_steps))
    assert len(observed_n_steps) == 1, 'ERROR: Parameter combinations must have the same number of runs.'
    observed_n_steps = observed_n_steps[0]

    return observed_n_steps


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
    p_value = (bs + 1) / (m + 1)
    return round(p_value, 15)


# ----------------------------------------------------------------------------------------------------------------------

st.title('Simulation Results – BigQuery')

# Select a particular simulation date and time
datasets = sorted([i.dataset_id for i in list(client.list_datasets())], reverse=True)
SIMULATION_ID  = st.sidebar.selectbox('Select simulation date.', options=datasets)
EXHIBIT = st.sidebar.selectbox('Select exhibit.', ['Static NetReward',
                                                   'Static Costs',
                                                   'Static Diagnostics',
                                                   'Dynamic NetReward',
                                                   'Dynamic Costs',
                                                   'Dynamic Diagnostics'])

os.makedirs(f'dyna-q/outputs/results/{SIMULATION_ID}', exist_ok=True)
sim_date_bq = SIMULATION_ID.replace('-', '_')
TABLE_NAME = f'`specialization-automation.{sim_date_bq}.results`'

config = yaml.load(open("config.yaml", 'r'), Loader=yaml.FullLoader)
run_results = run_query(f'select max(run_id) from {TABLE_NAME}')
RUNS = run_results[0]['f0_'] + 1

step_results = run_query(f'select max(orgstep_id) from {TABLE_NAME}')
STEPS = step_results[0]['f0_'] + 1

# -- Parse parameter values --------------------------------------------------------------------------------------------
lambda_results = run_query(f'select distinct lambda from {TABLE_NAME}')
LAMBDAs = sorted([i['lambda'] for i in lambda_results])

delta_results = run_query(f'select distinct delta from {TABLE_NAME}')
DELTAs  = sorted([i['delta'] for i in delta_results])

tau_results = run_query(f'select distinct tau from {TABLE_NAME}')
TAUs    = sorted([i['tau'] for i in tau_results])

lambda_ = st.sidebar.select_slider('Lambda: Specialization', options=LAMBDAs, value=2)
delta   = st.sidebar.select_slider('Delta: Frequency of environmental change', options=DELTAs, value=10)

parameters = run_query(f'select * from {TABLE_NAME} where lambda={lambda_} and delta={delta} limit 1;')

RUN_ALL_TABS = st.sidebar.checkbox('Run all queries')

# -- Plot aggregate results --------------------------------------------------------------------------------------------
if EXHIBIT == 'Static NetReward' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    # Execute queries
    validation_result = run_query(f'select count(row_index) from {TABLE_NAME} group by lambda, tau, delta, run_id')
    assert len(validation_result) == len(LAMBDAs) * len(DELTAs) * len(TAUs) * RUNS, f'ERROR: Validation of # of runs failed. {len(validation_result)}.'
    assert format_validation(validation_result) == STEPS, f'ERROR: Validation of # of steps failed.'
    rows = run_query(f'select lambda, tau, avg(sum_net_reward_to_org) from (select lambda, tau, sum(net_reward_to_org) as sum_net_reward_to_org from {TABLE_NAME} where delta={delta} and tau <> 1 group by lambda, tau, run_id) group by lambda, tau')

    # Accumulated net reward
    df = pd.DataFrame(rows).pivot(index='lambda', columns='tau')
    index_values = [f'lambda={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'tau={i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_netreward_avg_delta-{delta}_lambda-all_tau-all.csv')
    st.line_chart(df, height=500, width=500)
    st.dataframe(format_parameters(parameters, lambdas=LAMBDAs, taus=TAUs))
    st.write('Accumulated net rewards, averaged across simulation runs')
    st.table(df.round().astype(int).T)

    # Standard errors
    rows = run_query(f'select lambda, tau, STDDEV_SAMP(sum_net_reward_to_org) from (select lambda, tau, sum(net_reward_to_org) as sum_net_reward_to_org from {TABLE_NAME} where delta={delta} and tau <> 1 group by lambda, tau, run_id) group by lambda, tau')
    df = pd.DataFrame(rows).pivot(index='lambda', columns='tau')
    index_values = [f'lambda={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'tau={i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df = df / np.sqrt(RUNS-1)
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_netreward_std_delta-{delta}_lambda-all_tau-all.csv')
    st.write('Sample standard error with Bessel\'s correction')
    st.table(df.round(2).T)

    # Histogram
    st.write('-' * 3)
    st.subheader('Histogram: Distribution of net rewards')
    tau = st.select_slider('Select a single tau', TAUs, value=0)
    st.write('Percentile method, King, Tomz & Wittenberg (2000)')
    percentile_lower = st.number_input(label="Lower percentile", step=.05, value=0.05, format="%.2f")
    percentile_upper = st.number_input(label="Upper percentile", step=.05, value=0.95, format="%.2f")
    rows = run_query(f'select lambda, tau, sum(net_reward_to_org) as sum_net_reward_to_org from {TABLE_NAME} where delta={delta} and tau <> 1 group by lambda, tau, run_id')
    df = pd.DataFrame(rows)
    df = df[(df['tau'] == tau) & (df['lambda'] == lambda_)].copy()
    assert df.shape[0] == RUNS, f'ERROR: Expected {RUNS} observations, not {df.shape[0]}'
    plt.hist(df['sum_net_reward_to_org'], bins=100, color='black')
    plt.vlines(df['sum_net_reward_to_org'].mean(), ymin=0, ymax=100, color='red', label='Average')
    plt.vlines(df['sum_net_reward_to_org'].quantile(percentile_lower), ymin=0, ymax=100, color='green', linestyles='dashed', label=f'{percentile_lower}\% percentile')
    plt.vlines(df['sum_net_reward_to_org'].quantile(percentile_upper), ymin=0, ymax=100, color='green', linestyles='dashdot', label=f'{percentile_upper}\% percentile')
    plt.title(f'Histogram of accumulated net rewards, n={RUNS}\nlambda={lambda_}, delta={delta}, tau={tau}')
    plt.xlabel('Accumulated net reward')
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    # Jarque, Bera (1980) test for normality
    jb = stats.jarque_bera(df['sum_net_reward_to_org'].values)
    st.write(f'Jarque-Bera (1980) test for normality suggests normal distribution: {jb.pvalue > 0.05}')
    st.write(jb)

    # Shapiro-Wilk test for normality
    sw = stats.shapiro(df['sum_net_reward_to_org'].values)
    st.write(f'Shapiro-Wilk test for normality suggests normal distribution: {sw.pvalue > 0.05}')
    st.write(sw)

    # Permutation tests
    st.subheader('Permutation test')
    if st.button('Compute now'):

        st.write('Based on Ernst (2004), p. 682')

        rows = run_query(f'select lambda, tau, sum(net_reward_to_org) as sum_net_reward_to_org from {TABLE_NAME} where delta={delta} and tau <> 1 group by lambda, tau, run_id')
        df = pd.DataFrame(rows)
        assert df.shape[0] == len(LAMBDAs) * len(TAUs) * RUNS, f'ERROR: Expected {len(LAMBDAs) * len(TAUs) * RUNS} rows, not {df.shape[0]}'

        # Hold tau constant
        for t in TAUs:
            for j_focal, j_alter in itertools.combinations(LAMBDAs, 2):
                p_val = mc_perm_test(df[(df['tau'] == t) & (df['lambda'] == j_focal)]['sum_net_reward_to_org'].values,
                                     df[(df['tau'] == t) & (df['lambda'] == j_alter)]['sum_net_reward_to_org'].values, m=10000)
                st.write(f'lambda={j_focal} vs lambda={j_alter} | tau={round(t, 2)}: {p_val:.4f}')
            st.write('-' * 3)

        # Hold lambda constant
        for l in LAMBDAs:
            for j_focal, j_alter in itertools.combinations(TAUs, 2):
                p_val = mc_perm_test(df[(df['lambda'] == l) & (df['tau'] == j_focal)]['sum_net_reward_to_org'].values,
                                     df[(df['lambda'] == l) & (df['tau'] == j_alter)]['sum_net_reward_to_org'].values, m=10000)
                st.write(f'tau={round(j_focal, 2)} vs tau={round(j_alter, 2)} | lambda={l}: {p_val:.4f}')
            st.write('-' * 3)


if EXHIBIT == 'Static Costs' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    validation_result = run_query(f'select count(row_index) from {TABLE_NAME} where delta={delta} and lambda={lambda_} group by tau, run_id;')
    assert len(validation_result) == len(TAUs) * RUNS, f'ERROR: Validation of # of runs failed, {len(validation_result)}.'
    assert format_validation(validation_result) == STEPS, f'ERROR: Validation of # of runs failed, {format_validation(validation_result)}.'
    rows = run_query(f'select tau, sum(avg_coordination_costs), sum(avg_opportunity_costs) from (select tau, avg(coordination_costs) as avg_coordination_costs, avg(opportunity_costs) as avg_opportunity_costs from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau <> 1 group by tau, run_id) group by tau;')

    index_values = []
    coordination_costs = []
    opportunity_costs = []
    for row in rows:
        index_values.append('tau={}'.format(row['tau']))
        coordination_costs.append(row['f0_'])
        opportunity_costs.append(row['f1_'])
    df = pd.DataFrame({'coordination costs': coordination_costs, 'opportunity costs': opportunity_costs}, index=index_values)
    df.sort_index(inplace=True)

    plt.stackplot(df.index, df.T, labels=['Coordination costs', 'Opportunity costs'])
    plt.legend(df.columns, ncol=2)
    plt.xticks(ticks=range(0, len(df)), labels=[i[:8] for i in df.index])
    plt.ylabel('Accumulated costs across episodes')
    st.pyplot(plt)
    plt.clf()

    st.dataframe(format_parameters(parameters, taus=TAUs))

    st.table(df)
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_costs_sum_delta-{delta}_lambda-{lambda_}_tau-all.csv')

    # Coordination costs only for baseline - no auto, no env change
    rows = run_query(f'select lambda, avg(avg_coordination_costs) from (select lambda, sum(coordination_costs) as avg_coordination_costs from {TABLE_NAME} where delta=0 and tau=0 group by lambda, run_id) group by lambda;')
    df = pd.DataFrame(rows)
    df.columns = ['lambda', 'coordination_costs']
    df.sort_values(by='lambda', ascending=True, inplace=True)
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_coordination_avg_delta-0_lambda-all_tau-0.0.csv')

    # Dynamic learning
    # Coordination costs
    st.subheader('Coordination costs by episode and levels of automation')
    rows = run_query(f'select tau, orgstep_id, avg(coordination_costs) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau <> 1 group by tau, orgstep_id')
    df = pd.DataFrame(rows).pivot(index='tau', columns='orgstep_id')
    index_values = [f'tau={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'{i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df = df.T
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/dynamic_coordination_avg_delta-{delta}_lambda-{lambda_}_tau-0.0.csv')

    plt.plot(df.index.values.astype(int), df, alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df.columns], ncol=3)
    plt.xlabel('Episode')
    plt.ylabel('Coordination costs')
    plt.grid()
    plt.ylim(df.min().min()-1, df.max().max()+1)
    plt.ylim(-1, 2.5)
    st.pyplot(plt)
    plt.clf()

    st.write('Accumulated coordination costs, averaged across simulation runs')
    st.table(df.sum().round().astype(int))
    st.write('Average coordination costs, averaged across simulation runs')
    st.table(df.mean().round(4))


if EXHIBIT == 'Static Diagnostics' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    # Number of steps (aggregated)
    st.subheader('Average number of steps by levels of specialization and automation')
    validation_result = run_query(f'select count(row_index) as avg_moves from {TABLE_NAME} where delta={delta} group by lambda, tau, run_id;')
    assert len(validation_result) == len(TAUs) * len(LAMBDAs) * RUNS, f'ERROR: Validation of # of runs failed, {len(validation_result)}.'
    assert format_validation(validation_result) == STEPS, f'ERROR: Validation of # of runs failed, {format_validation(validation_result)}.'

    rows = run_query(f'select lambda, tau, avg(avg_moves) from (select lambda, tau, avg(moves) as avg_moves from {TABLE_NAME} where delta={delta} and tau <> 1 group by lambda, tau, run_id) group by lambda, tau')

    # Accumulated net reward
    df = pd.DataFrame(rows).pivot(index='lambda', columns='tau')
    index_values = [f'lambda={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'tau={i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_moves_avg_delta-{delta}_lambda-all_tau-all.csv')
    st.line_chart(df, height=500, width=500)
    st.dataframe(format_parameters(parameters, taus=TAUs, lambdas=LAMBDAs))
    st.table(df.round(2).T)


if EXHIBIT == 'Dynamic NetReward' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    rows = run_query(f'select tau, orgstep_id, avg(net_reward_to_org) from {TABLE_NAME} where delta={delta} and lambda={lambda_} group by tau, orgstep_id')
    validation_result = run_query(f'select count(row_index) from {TABLE_NAME} where delta={delta} and lambda={lambda_} group by tau, orgstep_id')
    assert len(validation_result) == len(TAUs) * STEPS, f'ERROR: Validation of # of steps failed, {len(validation_result)}.'
    assert format_validation(validation_result) == RUNS, f'ERROR: Validation of # of runs failed, {format_validation(validation_result)}.'

    df = pd.DataFrame(rows).pivot(index='tau', columns='orgstep_id')
    index_values = [f'tau={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'{i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df = df.T
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/dynamic_netreward_avg_delta-{delta}_lambda-{lambda_}_tau-all.csv')

    plt.plot(df.index.values.astype(int), df, alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df.columns], ncol=3)
    plt.xlabel('Episode')
    plt.ylabel('Average net reward')
    plt.grid()
    plt.ylim(df.min().min()-2, df.max().max()+1)
    st.pyplot(plt)
    plt.clf()

    # Accumulated
    df_cumsum = df.cumsum()
    plt.plot(df_cumsum.index.values.astype(int), df_cumsum, alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df_cumsum.columns], ncol=3)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative average net reward')
    plt.grid()
    st.pyplot(plt)
    plt.clf()

    st.dataframe(format_parameters(parameters, taus=TAUs))

    st.write('Accumulated net rewards, averaged across simulation runs')
    st.table(df.sum().round().astype(int))

    st.write('Average net rewards, averaged across simulation runs')
    st.table(df.mean().round(2))

if EXHIBIT == 'Dynamic Costs' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    # Average opportunity costs (dynamic learning)
    st.subheader('Opportunity costs by episode and levels of automation')
    rows = run_query(f'select tau, orgstep_id, avg(opportunity_costs) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau <> 1 group by tau, orgstep_id')

    df = pd.DataFrame(rows).pivot(index='tau', columns='orgstep_id')
    index_values = [f'tau={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'{i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df = df.T

    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/dynamic_opportunity_avg_delta-{delta}_lambda-{lambda_}_tau-all.csv')
    df.cumsum().to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/static_opportunity_avg_delta-{delta}_lambda-{lambda_}_tau-all.csv')

    # Averaged
    plt.plot(df.index.values.astype(int), df, alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df.columns], ncol=4)
    plt.xlabel('Episode')
    plt.ylabel('Average opportunity costs')
    plt.grid()
    plt.ylim(df.min().min()-0.5, df.max().max()+2)
    st.pyplot(plt)
    plt.clf()

    # Accumulated
    plt.plot(df.index.values.astype(int), df.cumsum(), alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df.columns], ncol=4)
    plt.xlabel('Episode')
    plt.ylabel('Accumulated opportunity costs')
    plt.grid()
    st.pyplot(plt)
    plt.clf()

    st.dataframe(format_parameters(parameters, taus=TAUs))

    st.write('Accumulated opportunity costs, averaged across simulation runs.')
    st.table(df.sum().round().astype(int))
    st.write('Average opportunity costs, averaged across simulation runs.')
    st.table(df.mean().round(4))


if EXHIBIT == 'Dynamic Diagnostics' or RUN_ALL_TABS:
    st.title(EXHIBIT)

    # Number of steps (dynamic learning)
    st.subheader('Average number of steps by episode and levels of automation')
    rows = run_query(f'select tau, orgstep_id, avg(moves) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau <> 1 group by tau, orgstep_id')
    validation_result = run_query(f'select count(row_index) from {TABLE_NAME} where delta={delta} and lambda={lambda_} group by tau, orgstep_id')
    assert len(validation_result) == len(TAUs) * STEPS, f'ERROR: Validation of # of steps failed, {len(validation_result)}.'
    assert format_validation(validation_result) == RUNS, f'ERROR: Validation of # of runs failed, {format_validation(validation_result)}.'

    df = pd.DataFrame(rows).pivot(index='tau', columns='orgstep_id')
    index_values = [f'tau={i}' for i in df.index]
    column_values = [i[1] for i in df.columns]
    column_names = [f'{i}' for i in column_values]
    df.index = index_values
    df.columns = column_names
    df = df.T
    df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/dynamic_moves_avg_delta-{delta}_lambda-{lambda_}_tau-all.csv')

    plt.plot(df.index.values.astype(int), df, alpha=0.8)
    plt.xticks(ticks=[i for i in range(STEPS+1) if i % 100 == 0], rotation=90)
    plt.legend([i[:8] for i in df.columns], ncol=4)
    plt.xlabel('Episode')
    plt.ylabel('Number of steps')
    plt.grid()
    plt.ylim(df.min().min()-0.5, df.max().max()+2)
    st.pyplot(plt)
    plt.clf()

    st.dataframe(format_parameters(parameters, taus=TAUs))

    st.write('Accumulated number of steps, averaged across simulation runs.')
    st.table(df.sum().round().astype(int))
    st.write('Average number of steps, averaged across simulation runs.')
    st.table(df.mean().round(4))

    # Path taken
    st.subheader('Path taken')
    tau = st.select_slider('Select a single tau', TAUs, value=0, key='tau1')
    rows = run_query(f'select path_taken, orgstep_id, COUNT(*) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau={tau} group by path_taken, orgstep_id')
    df = pd.DataFrame(rows)
    df = pd.get_dummies(df, columns=['path_taken'], drop_first=False)
    df['path_taken_short path (closed door)'] = df['path_taken_short path (closed door)'] * df['f0_']
    df['path_taken_long path'] = df['path_taken_long path'] * df['f0_']
    df['path_taken_short path (open door)'] = df['path_taken_short path (open door)'] * df['f0_']
    df.drop(labels=['f0_'], axis=1, inplace=True)
    df = df.groupby(by='orgstep_id').sum()/RUNS

    plt.bar(range(STEPS), df['path_taken_short path (closed door)'], label='Closed door')
    plt.bar(range(STEPS), df['path_taken_long path'], bottom=df['path_taken_short path (closed door)'], label='Exploitation path')
    plt.bar(range(STEPS), df['path_taken_short path (open door)'], bottom=df['path_taken_short path (closed door)'] + df['path_taken_long path'], label='Exploration path')
    plt.xlabel('Episode')
    plt.ylabel('Share')
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    # Loop through all taus for writing data to disk
    for t in TAUs:
        rows = run_query(f'select path_taken, orgstep_id, COUNT(*) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau={t} group by path_taken, orgstep_id')
        df = pd.DataFrame(rows)
        df = pd.get_dummies(df, columns=['path_taken'], drop_first=False)
        df['path_taken_short path (closed door)'] = df['path_taken_short path (closed door)'] * df['f0_']
        df['path_taken_long path'] = df['path_taken_long path'] * df['f0_']
        df['path_taken_short path (open door)'] = df['path_taken_short path (open door)'] * df['f0_']
        df.drop(labels=['f0_'], axis=1, inplace=True)
        df = df.groupby(by='orgstep_id').sum() / RUNS
        df.to_csv(f'dyna-q/outputs/results/{SIMULATION_ID}/dynamic_paths_count_delta-{delta}_lambda-{lambda_}_tau-{t}.csv')

    # Leading agent
    st.subheader('Leading agent share by episode')
    tau = st.select_slider('Select a single tau', TAUs, value=0, key='tau2')
    rows = run_query(f'select leader, orgstep_id, COUNT(*) from {TABLE_NAME} where delta={delta} and lambda={lambda_} and tau={tau} group by leader, orgstep_id')

    df = pd.DataFrame(rows)
    df.sort_values(by='orgstep_id', ascending=True, inplace=True)
    assert len(df) >= STEPS * lambda_ or tau == 1, f'ERROR: Check size of leading agent data, now {df.shape}'

    # Construct plottable data
    data = []
    labels = []
    for leader in set(df['leader'].values):
        d = np.zeros(STEPS)
        for s in range(STEPS):
            leader_count = df[(df['leader'] == leader) & (df['orgstep_id'] == s)]['f0_']
            if len(leader_count) == 1:
                d[s] = leader_count
            else:
                pass

        data.append(d)
        labels.append(leader.replace('_', '-'))

    plt.stackplot(range(STEPS), data, labels=labels)
    plt.xlabel('Episode')
    plt.ylabel('Runs')
    plt.legend()
    st.pyplot(plt)
    plt.clf()