""" Utilities to support running Google Cloud VMs"""

import subprocess
import time
import datetime as dt

import urllib.request
from socket import gethostname

F_STARTUP_SCRIPT       = 'rpc.sh'
GIT_URL                = 'https://<username>:<token>@github.com/mxhofer/OrgSim-RL.git'
GC_PROJECT             = '<project_id>'
GC_IMAGE               = '<image_name>'
GC_ZONE                = 'us-central1-a'


def vm_start_instance(vmname, rpc_cd='', rpc_run='', vmtype='c2-standard-4', giturl='', debug=False, preemptible=False):
    """
    Start a Google VM
    :param vmname: name of the vm instance in Google Compute
    :param rpc_cd: A bash command to change directory (e.g. "cd my_dir")
    :param rpc_run: A bash command to set all internal parameters and run the script
                        (e.g. "sudo python my_script.py tau 0.1 delta 5 lambda 2")
    :param vmtype: One of the predefined Google Compute instance types
    :param giturl: Clone the GitHub repo at this URL on startup. Include username and token for private repos.
    :param preemptible: If True, will start VM in preemptible mode
    """
    assert vmname

    if debug:
        rpc_run += ' debug'
    bash_line = ['gcloud', 'compute', 'instances', 'create', vmname]
    bash_line.extend(['--project',            GC_PROJECT])
    bash_line.extend(['--zone',               GC_ZONE])
    bash_line.extend(['--image',              GC_IMAGE])
    bash_line.extend(['--machine-type',       vmtype])
    if preemptible:
        bash_line.extend(["--preemptible"])
    if F_STARTUP_SCRIPT:
        bash_line.extend(['--metadata-from-file', 'startup-script=' + F_STARTUP_SCRIPT])
    bash_line.extend(['--metadata',           'rpc_cd=' + rpc_cd + ',rpc_run=' + rpc_run + ',giturl=' + giturl])
    bash_line.extend(['--network',            'default', '--no-restart-on-failure', '--maintenance-policy', 'TERMINATE',
                      '--scopes',             'https://www.googleapis.com/auth/userinfo.email,' +
                                              'https://www.googleapis.com/auth/compute,' +
                                              'https://www.googleapis.com/auth/devstorage.full_control,' +
                                              'https://www.googleapis.com/auth/taskqueue,' +
                                              'https://www.googleapis.com/auth/bigquery,' +
                                              'https://www.googleapis.com/auth/sqlservice.admin,' +
                                              'https://www.googleapis.com/auth/datastore,' +
                                              'https://www.googleapis.com/auth/cloud-platform,' +
                                              'https://www.googleapis.com/auth/projecthosting'
                    ]
                   )
    print('Starting  ' + str(vmname) +
          '  ' + vmtype +
          '  ' + rpc_cd +
          '  ' + rpc_run +
          '  ' + ("preemptible" if preemptible else "non-preemptible"))
    subprocess.Popen(bash_line)


def validate_outputs(start_date):
    """
    Validate that all output files are writte to the bucket.
    :param start_date: simulation start date
    """

    # Expected files
    expected_files = []
    for t in TAUs:
        for d in DELTAS:
            for l in LAMBDAS:
                expected_files.append(f'tau_{t}-delta_{d}-lambda_{l}')

    # Check against actual files
    cmd = f'gsutil ls -r gs://simulation-output/results/{start_date}/'
    actual_files = subprocess.check_output(cmd.split(' ')).decode("utf-8").split('\n')
    actual_files = list(set([i for i in actual_files if '.csv' in i]))
    missing_files = []
    matching_files = 0
    for e_file in expected_files:
        file_found = False
        for a_file in actual_files:
            if e_file in a_file:
                matching_files += 1
                file_found = True
        if not file_found:
            missing_files.append(e_file)

    missing_files = '\n\t'.join(missing_files)

    print(f'Expected files: {len(expected_files)}')
    print(f'Actual files: {len(actual_files)}')
    print(f'Matching files: {matching_files}')
    print(f'Missing files: \n\t{missing_files}')


if __name__ == '__main__':

    # Set parameter ranges
    TAUs = [0, 0.5, 0.95]
    DELTAS = [0, 10, 1000]
    LAMBDAS = [1, 2, 4]

    START_TIME = dt.datetime.now().strftime('%H-%M-%S')

    # Launch VMs
    for tau in TAUs:
        for delta in DELTAS:
            for lambda_ in LAMBDAS:
                rpc_run = f'sudo /opt/conda/bin/python3 dynaQ.py tau {tau} delta {delta} lambda {lambda_}'

                # Make sure the GitHub access token in the 'giturl' parameter is up-to-date
                vm_start_instance(vmname='vm-{}-tau-{}-delta-{}-lambda-{}'.format(START_TIME, str(tau).replace('.', ''), delta, lambda_),
                                  rpc_cd='cd automation/dyna-q/',
                                  rpc_run=rpc_run,
                                  giturl=GIT_URL)

                # Sleep for five seconds
                time.sleep(5)

    # Validate the results in the Google Storage bucket
    # validate_outputs('2022-01-13')

