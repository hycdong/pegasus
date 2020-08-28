#!/bin/bash

PID=$$

function usage()
{
    echo "This tool is for bulk load specified table(app)."
    echo "USAGE: $0 -h -m --meta_list -a app_name -c cluster_name -p provider_name"
    echo "Options:"
    echo "  -h|--help                       print help message"
    echo
    echo "  -m|--meta_list <str>            cluster meta server list"
    echo
    echo "  -a|--app_name <str>             target table(app) name"
    echo
    echo "  -c|--cluster_name <str>         target cluster name"
    echo
    echo "  -p|--provider_name <str>        target provider name"
    echo
    echo "  --not_update_rocksdb_scenario   not update rocksdb scenario during bulk load"
    echo "                                  this option should only be used when no write request during bulk load"
    echo
    echo "  --not_trigger_manual_compact    not trigger manual compaction after bulk load"
    echo
    echo "  --max_concurrent_manual_compact_running_count <num>"
    echo "                                  max concurrent manual compaction running count limit, should be positive integer."
    echo "                                  if not set, default value is 1"
    echo
    echo "for example:"
    echo
    echo "Start bulk load:"
    echo
    echo "      $0 -c 127.0.0.1:34601,127.0.0.1:34602 -a temp -c cluster -p local_service"
    echo
}

# set_env meta_list app_name key value
function set_env()
{
    meta_list=$1
    app_name=$2
    key=$3
    value=$4

    log_file="/tmp/$UID.$PID.pegasus.set_app_envs.${app_name}"
    echo -e "use ${app_name}\n set_app_envs ${key} ${value}" | ./run.sh shell --cluster ${meta_list} &>${log_file}
    set_fail=`grep 'set app env failed' ${log_file} | wc -l`
    if [ ${set_fail} -eq 1 ]; then
        echo "ERROR: set app envs failed, refer to ${log_file}"
        exit 1
    fi
    echo -e "set app(${app_name}) env ${key} = ${value}\n"
}

# start_bulk_load meta_list app_name cluster_name provider_name
function start_bulk_load()
{
    meta_list=$1
    app_name=$2
    cluster_name=$3
    provider_name=$4

    start_cmd="start_bulk_load -a ${app_name} -c ${cluster_name} -p ${provider_name}"
    log_file="/tmp/$UID.$PID.pegasus.start_bulk_load.${app_name}"
    echo -e "${start_cmd}" | ./run.sh shell --cluster ${meta_list} &>${log_file}
    start_fail=`grep 'failed' ${log_file} | wc -l`
    if [ ${start_fail} -eq 1 ]; then
        echo "ERROR: start bulk load failed, refer to ${log_file}"
        if [ "${need_update_scenario}" == "true" ]; then
            set_env ${meta_list} ${app_name} "rocksdb.usage_scenario" "normal"
        fi
        exit 1
    fi
}

if [ $# -eq 0 ]; then
    usage
    exit 0
fi

# parse parameters
meta_list=""
app_name=""
cluster_name=""
provider_name=""
need_update_scenario="true"
need_compact="true"
compact_running_count="1"
while [[ $# > 0 ]]; do
    option_key="$1"
    case ${option_key} in
        -m|--meta_list)
            meta_list="$2"
            shift
            ;;
        -a|--app_name)
            app_name="$2"
            shift
            ;;
        -c|--cluster_name)
            cluster_name="$2"
            shift
            ;;
        -p|--provider_name)
            provider_name="$2"
            shift
            ;;
        --not_update_rocksdb_scenario)
            need_update_scenario="false"
            ;;
        --not_trigger_manual_compact)
            need_compact="false"
            ;;
        --max_concurrent_manual_compact_running_count)
            compact_running_count="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
    esac
    shift
done

# cd to shell dir
pwd="$(cd "$(dirname "$0")" && pwd)"
shell_dir="$(cd ${pwd}/.. && pwd )"
cd ${shell_dir}

# check meta_list
if [ "${meta_list}" == "" ]; then
    echo "ERROR: invalid meta_list: ${meta_list}"
    exit 1
fi

# check app_name
if [ "${app_name}" == "" ]; then
    echo "ERROR: invalid app_name: ${app_name}"
    exit 1
fi

# check cluster_name
if [ "${cluster_name}" == "" ]; then
    echo "ERROR: invalid cluster_name: ${cluster_name}"
    exit 1
fi

# check provider_name
if [ "${provider_name}" == "" ]; then
    echo "ERROR: invalid provider_name: ${provider_name}"
    exit 1
fi

# check compact_running_count
if [ "${need_compact}" == "true" ]; then
    expr ${compact_running_count} + 0 &>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: invalid compact_running_count: ${compact_running_count}"
        exit 1
    fi
    if [ ${compact_running_count} -lt 0 ]; then
        echo "ERROR: invalid compact_running_count: ${compact_running_count}"
        exit 1
    fi
fi


# record start time
start_time=`date +%s`
echo "UID: $UID"
echo "PID: $PID"
echo "meta_list: $meta_list"
echo "app_name: $app_name"
echo "cluster_name: $cluster_name"
echo "provider_name: $provider_name"
echo "Start time: `date -d @${start_time} +"%Y-%m-%d %H:%M:%S"`"
echo

if [ "${need_update_scenario}" == "true" ]; then
    # set app rocksdb usage_scenario as bulk_load to avoid write stalling
    set_env ${meta_list} ${app_name} "rocksdb.usage_scenario" "bulk_load"
fi

# start bulk load
start_bulk_load_time=`date +%s`
start_bulk_load ${meta_list} ${app_name} ${cluster_name} ${provider_name}
echo "=================================================================="
echo "Starting bulk load"
# wait bulk load finished
query_cmd="query_bulk_load_status -a ${app_name}"
last_bulk_load_status=""
start_downloading_time=""
start_ingestion_time=""

slept=0
while true
do
    log_file="/tmp/$UID.$PID.pegasus.query_bulk_load.${app_name}"
    echo "${query_cmd}" | ./run.sh shell --cluster ${meta_list} &>${log_file}

    query_finish=`grep 'not during bulk load' ${log_file} | wc -l`
    if [ ${query_finish} -eq 1 ]; then
        break
    fi

    app_bulk_load_status=`grep 'app_bulk_load_status' ${log_file} | awk '{print $3}'`
    if [ "${app_bulk_load_status}" == "BLS_DOWNLOADING" ]; then
        if [ "${last_bulk_load_status}" == "" ]; then
            start_downloading_time=`date +%s`
            echo "[${slept}s] start downloading files [`date -d @${start_downloading_time} +"%Y-%m-%d %H:%M:%S"`]"
        else
            download_progress=`grep 'app_total_download_progress' ${log_file} | awk '{print $3}'`
            echo "[${slept}s] downloading files, download progress = ${download_progress}"
        fi
    fi

    if [ "${app_bulk_load_status}" == "BLS_DOWNLOADED" ]; then
        if [ "${last_bulk_load_status}" == "BLS_DOWNLOADING" ]; then
            download_finish_time=`date +%s`
            echo "[${slept}s] download files succeed, elapsed time is $((download_finish_time - start_downloading_time)) seconds."
        else
            echo "[${slept}s] waiting for starting ingestion"
        fi
    fi

    if [ "${app_bulk_load_status}" == "BLS_INGESTING" ]; then
        if [ "${last_bulk_load_status}" == "BLS_DOWNLOADED" ]; then
            start_ingestion_time=`date +%s`
            echo "[${slept}s] start ingestion files, reject write requests [`date -d @${start_ingestion_time} +"%Y-%m-%d %H:%M:%S"`]"
        else
            echo "[${slept}s] waiting for ingestion finish"
        fi
    fi

    if [ "${app_bulk_load_status}" == "BLS_SUCCEED" ]; then
        if [ "${last_bulk_load_status}" == "BLS_INGESTING" ]; then
            ingestion_finish_time=`date +%s`
            echo "[${slept}s] ingestion files succeed, recover write, elapsed time is $((ingestion_finish_time - start_ingestion_time)) seconds."
        else
            echo "[${slept}s] waiting for cleaning up bulk load context"
        fi
    fi

    if [ "${app_bulk_load_status}" == "BLS_FAILED" ]; then
        if [ "${last_bulk_load_status}" != "BLS_FAILED" ]; then
            echo "[${slept}s] bulk load failed, old status is ${last_bulk_load_status}"
        else
            echo "[${slept}s] waiting for cleaning up bulk load context"
        fi
    fi

    last_bulk_load_status=${app_bulk_load_status}

    sleep 10
    slept=$((slept + 10))
done

finish_status="done"
if [ "${last_bulk_load_status}" == "BLS_SUCCEED" ]; then
    finish_status="succeed"
elif [ "${last_bulk_load_status}" == "BLS_FAILED" ]; then
    finish_status="failed"
    need_compact="false"
fi

# record finish time
bulk_load_finish_time=`date +%s`
echo
echo "Finish time: `date -d @${bulk_load_finish_time} +"%Y-%m-%d %H:%M:%S"`"
echo "Bulk Load ${finish_status}, elapsed time is $((bulk_load_finish_time - start_bulk_load_time)) seconds."
echo

if [ "${need_compact}" == "true" ]; then
    echo "=================================================================="
    echo "Starting manual compact"
    source ./scripts/pegasus_manual_compact.sh -c ${meta_list} -a ${app_name} --max_concurrent_running_count ${compact_running_count}
    echo
fi

if [ "${need_update_scenario}" == "true" ]; then
    # recover app rocksdb usage_scenario as normal after manual compaction
    set_env ${meta_list} ${app_name} "rocksdb.usage_scenario" "normal"
fi

if [ "${need_compact}" == "true" ]; then
    # record finish time
    total_finish_time=`date +%s`
    echo
    echo "Finish time: `date -d @${total_finish_time} +"%Y-%m-%d %H:%M:%S"`"
    echo "Bulk load and manual compaction done, elapsed time is $((total_finish_time - start_time)) seconds."
    echo
fi

rm -f /tmp/$UID.$PID.pegasus.* &>/dev/null
