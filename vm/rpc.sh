#! /bin/bash

#=======================================================================================================================

echo '~'
echo '~'
echo '~'
echo '|||||||||||||||||||||||||||||||||||'
echo '|||||||||||||||||||||||||||||||||||'
echo '            RPC STARTUP            '
echo '|||||||||||||||||||||||||||||||||||'
echo '|||||||||||||||||||||||||||||||||||'
echo '~'
echo '~'
echo '~'
echo '~'

#=======================================================================================================================

echo '-----------------------------------'
echo '            GIT CLONE              '
echo '-----------------------------------'
echo ' '
GITURL=$(curl -sS http://metadata/computeMetadata/v1/instance/attributes/giturl -H "Metadata-Flavor: Google")
cd /
sudo git clone "${GITURL}"
success=$?
echo "$success"
if [[ ${success} -eq 0 ]];
then
    echo '~'
    echo '-----------------------------------'
    echo '        GIT CLONE COMPLETED        '
    echo '-----------------------------------'
    echo '~'
else
    echo "error: git pull failed - will sleep 30 seconds and retry"
    sleep 30s
    sudo git clone "${GITURL}"
    success=$?
    if [[ ${success} -eq 0 ]];
    then
        echo '~'
        echo '-----------------------------------'
        echo '        GIT CLONE COMPLETED        '
        echo '-----------------------------------'
        echo '~'
    else
        echo "error: git pull failed - will sleep 90 seconds and retry"
        sleep 90s
        sudo git clone "${GITURL}"
        success=$?
        if [[ ${success} -eq 0 ]];
        then
            echo '~'
            echo '-----------------------------------'
            echo '        GIT CLONE COMPLETED        '
            echo '-----------------------------------'
            echo '~'
        else
            echo '***********************************'
            echo '***********************************'
            echo "   GIT CLONE FAILED - NO RETRIES   "
            echo '***********************************'
            echo '***********************************'
        fi
    fi
fi

#=======================================================================================================================

echo '~'
echo '-----------------------------------'
echo '            EXECUTE RPC            '
echo '-----------------------------------'
echo '~'
RPC_CD=$(curl -sS http://metadata/computeMetadata/v1/instance/attributes/rpc_cd -H "Metadata-Flavor: Google")
RPC_RUN=$(curl -sS http://metadata/computeMetadata/v1/instance/attributes/rpc_run -H "Metadata-Flavor: Google")
if [[ -n "$RPC_CD" ]];
then
    echo "${RPC_CD}"
    ${RPC_CD}

    echo "${RPC_RUN}"
    ${RPC_RUN}

    if [[ $? = 0 ]]; then
        echo '~'
        echo '-----------------------------------'
        echo '             RPC DONE              '
        echo '-----------------------------------'
        echo '~'

    else
        echo '~'
        echo '~'
        echo '***********************************'
        echo '***********************************'
        echo '            RPC CRASHED            '
        echo '***********************************'
        echo '***********************************'
    fi
fi

#=======================================================================================================================

echo '~'
echo '|||||||||||||||||||||||||||||||||||'
echo '|||||||||||||||||||||||||||||||||||'
echo '             HIBERNATE             '
echo '|||||||||||||||||||||||||||||||||||'
echo '|||||||||||||||||||||||||||||||||||'
echo '~'
systemctl hibernate
