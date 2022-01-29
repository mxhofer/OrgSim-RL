# AutoOrgRL Project

Copyright (c) 2022 Maximilian W. Hofer & Kenneth A. Younge.

Repo maintainer: Maximilian W. Hofer ([maximilian.hofer@epfl.ch](mailto:maximilian.hofer@epfl.ch))

	AUTHOR:  Maximilian W. Hofer  
	SOURCE:  https://github.com/mxhofer/OrgSim-RL  
	LICENSE: Access to this code is provided under an MIT License.  

The AutoOrgRL project investigates how automation can impact organizations using a reinforcement learning simulation.

# Key directories and files

- `dyna-q/`: dyna-Q algorithm 
- `vm/`: virtual machine scripts
- `.streamlit/`: Streamlit authentication
- `config.yaml`: all parameter configurations
- `dashboard.py`: Streamlit dashboard

# Usage
## Fork repo

Create your own copy of the repository to freely experiment with AutoOrgRL.

## Install dependencies

`git clone https://github.com/mxhofer/OrgSim-RL.git`

`pip install -r requirements.txt`

## Prepare Google Cloud environment

You will need to enable the following services:

- BigQuery
- AppEngine (make sure you have deployment rights)
- Container Registry 
- IAM (make sure you have permission to create a new service account) 

## Configure the virtual machine (VM)

1. Create a VM with a disk (e.g. 200GB)
2. Add to the disk: 
   1. `vm/rpc.sh` script
   2. When you stop this first VM, keep the disk around so you can re-use copies of the disk for future VMs
3. Stop the VM
4. Create a new image from the disk. Future VMs will use a copy of this image.
5. Update VM configurations in `vm/vm.py`:
   1. GitHub URL
      1. Format: `https://<username>:<token>@<github_url>`
   2. Project name
   3. Image name
   4. Compute zone 

## Run simulation

The `config.yaml` file contains all parameter values. The `vm/vm.py` file starts VMs overwriting parameters values for automation (tau), environmental change (delta), and specializaiton (lambda).

1. Set parameters in `config.yaml` 
2. Set a value for the `TAG` variable at the top of `dyna-q/dynaQ.py` to identify the simulation run.
3. Push repository to GitHub 
4. Set parameter ranges in `vm/vm.py`
5. Run `vm/vm.py`
6. Go to Google Cloud / Compute Engine / VM instances
   1. Click on a VM
   2. Open `Serial Port 1 (console)` to check stdout log
7. Check Google Cloud / Cloud Storage / Browser for simulation ouputs:
   1. Check bucket `simulation-output/results/`
8. Validate outputs
   1. Use the `validate_outputs` function in `vm/vm.py`

## Ingest results into BigQuery

1. Follow the general guide on ingesting .csv files into BigQuery [here](https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-csv#console)
   1. Data set ID: date + TAG
   2. Create table from: select a .csv output file in the appropriate bucket. Then edit the filename to `*.csv` to ingest all files in that bucket.
   3. Table name: `results` (hard requirement as the SQL queries expect this table name!)
   4. Schema: Auto detect.
   5. Cluster by `delta, lambda` to speed up querying performance.
   6. In Advanced Options. Header rows to skip: 1
   7. Create table.
2. Validate number of rows in table
   1. Click on the table name
   2. Navigate to `Details`
   3. Check `Number of rows`. Number of rows should equal:
      1. \# of values for lambda x \# of values for tau x \# of values lambda x \# episodes x \# runs 

## Configure Streamlit

1. On Google Cloud: 
   1. Create a service account with Viewer permissions. See details [here](https://docs.streamlit.io/knowledge-base/tutorials/databases/bigquery).
   2. Create a key. 
   3. Download the key as a JSON. 
2. Save the key file in `.streamlit/`
3. Update the path to the key file in `dashboard.py`
4. Add the key file to `.gitignore`

## Deploy dashboard

1. Test that the dashboard is running locally:
   1. `streamlit run dashboard.py --server.port=8080 --server.address=0.0.0.0`
   2. Running the dashboard will write .csv files of the outputs to disk in `dyna-q/outputs`
2. Test that the dashboard is running locally in a Docker container:
   1. `docker build . -t dashboard`
   2. `docker run -p 8080:8080 dashboard`
3. Deploy dashboard:
   1. `gcloud app deploy dashboard.yaml`
   2. Click on the URL of the deployed service

