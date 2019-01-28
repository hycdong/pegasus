#!/bin/bash

PID=$$

function usage()
{
    echo
    echo "This tool is for partition specified table(app)."
    echo "USAGE: $0 -c cluster -a app_name -p partition_count -m max_memused_radio [-mr max_memory_used_radio][-dr max_disk_used_radio] [...]"
    echo "Options:"
    echo "  -h|--help                           print help message"
    echo
    echo "  -c|--cluster <str>                  cluster meta server list"
    echo
    echo "  -a|--app_name <str>                 target table name"
    echo
    echo "  -p|--partition_count <num>          new partition count, should be double of original partition_count"
    echo "                                      e.g. original partition_count=4, new partition_count must be 8"
    echo
    echo "  -m|--max_memused_radio <num>        the max memory used precent of all replica server"
    echo "                                      we can get from monitor screen or pysical machines"
    echo "                                      e.g. replica1 20%, replica2 30%, replica3 25%, this option should be 30"
    echo
    echo "  -mr|--max_memory_used_radio <num>   memory_used_max_ratio after split the table, default is 80"
    echo "                                      partition split will increase memory, we should check memory below the thereshold"
    echo
    echo "  -mr|--max_disk_used_radio <num>     disk_used_max_ratio after split the table, default is 80"
    echo "                                      partition split will increase disk usage, we should check disk below the thereshold"
    echo
}

function check_percent()
{
    param_value=$1
    param_msg=$2

    expr ${param_value} + 0 &>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: invalid ${param_msg}: ${param_value}"
        exit 1
    fi
    if [ ${param_value} -lt 0 ]; then
        echo "ERROR: invalid ${param_msg}: ${param_value}"
        exit 1
    fi
    if [ ${param_value} -gt 100 ]; then
        echo "ERROR: invalid ${param_msg}: ${param_value}"
        exit 1
    fi
}

if [ $# -eq 0 ]; then
    usage
    exit 0
fi

# parse parameters
cluster=""
app_name=""
partition_count=""
max_memory_used_percent=""
disk_max_used_percent_after_split=80
memory_max_used_percent_after_split=80

while [[ $# -gt 0 ]]; do
    case "$1" in
        "-c"|"--cluster")
            cluster="$2"
            shift
            ;;
        "-a"|"--app_name")
            app_name="$2"
            shift
            ;;
        "-p"|"--partition_count")
            partition_count="$2"
            shift
            ;;
        "-m"|"--max_memused_radio")
            max_memory_used_percent="$2"
            shift
            ;;
        "-dr"|"--max_disk_used_radio")
            disk_max_used_percent_after_split="$2"
            shift
            ;;
        "-mr"|"--max_memory_used_radio")
            memory_max_used_percent_after_split="$2"
            shift
            ;;
        "-h"|"--help")
            usage
            exit 0
            ;;
    esac
    shift
done

# check parameters
if [ "${cluster}" == "" ]; then
    echo "ERROR: invalid cluster ${cluster}"
    exit 1
fi

if [ "${app_name}" == "" ]; then
    echo "ERROR: invalid app_name ${app_name}"
    exit 1
fi

expr ${partition_count} + 0 &>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: invalid partition_count: ${partition_count}"
    exit 1
fi
if [ ${partition_count} -lt 0 ]; then
    echo "ERROR: invalid partition_count: ${partition_count}"
    exit 1
fi

check_percent ${max_memory_used_percent} "replica server current max memory used percent"
check_percent ${disk_max_used_percent_after_split} "replica server max disk used percent after split"
check_percent ${memory_max_used_percent_after_split} "replica server max memory used percent after split"

if [ ${disk_max_used_percent_after_split} -lt 80 ]; then
    disk_max_used_percent_after_split=80
fi
if [ ${memory_max_used_percent_after_split} -lt 80 ]; then
    memory_max_used_percent_after_split=80
fi

# cd to shell dir
script_dir="$(cd "$(dirname "$0")" && pwd)"
shell_dir="$(cd "${script_dir}/.." && pwd)"
cd ${shell_dir}

all_start_time=`date +%s`
echo "UID: $UID"
echo "PID: $PID"
echo "cluster: $cluster"
echo "app_name: $app_name"
echo "new partition_count: $partition_count"
echo "current max memory used percent: $max_memory_used_percent"
echo "replica server max disk used percent after split: $disk_max_used_percent_after_split"
echo "replica server max memory used percent after split: $memory_max_used_percent_after_split"
echo "Start time: `date -d @${all_start_time} +"%Y-%m-%d %H:%M:%S"`"
echo

# check app stats and partition_count
original_pc=""
ls_log_file="/tmp/$UID.$PID.pegasus.ls"
echo ls | ./run.sh shell --cluster ${cluster} &>${ls_log_file}
while read app_line
do
    app_id=`echo ${app_line} | awk '{print $1}'`
    status=`echo ${app_line} | awk '{print $2}'`
    app=`echo ${app_line} | awk '{print $3}'`
    pc=`echo ${app_line} | awk '{print $5}'`

    if [ "${app}" != "${app_name}" ]; then
        continue
    fi

    if [ "${status}" != "AVAILABLE" ]; then
        echo "ERROR: app ${app_name} is not available now, please query later"
        exit 1
    fi

    original_pc=${pc}
    correct_pc=$[ ${pc} * 2 ]
    if [ ${correct_pc} != ${partition_count} ]; then
        echo "ERROR: wrong partition_count ${partition_count}, should be ${correct_pc}"
        exit 1
    fi
done<${ls_log_file}

# check app partition healthy
full_healthy_count=`echo "app ${app_name} -d" | ./run.sh shell --cluster ${cluster} | grep "fully_healthy_partition_count" | awk '{print $3}'`
if [ ${full_healthy_count} != ${original_pc} ]; then
    echo "ERROR: app ${app_name} unhealthy partition count is $((original_pc - full_healthy_count)), please query later"
    exit 1
fi

# check available memory and disk
max_memory_used_MB=0
max_disk_used_ratio=0
memory_total_MB=131072 #128G
disk_total_MB=0
server_stat_file="/tmp/$UID.$PID.pegasus.server_stat"
echo "server_stat -t replica-server" | ./run.sh shell --cluster ${cluster} &>${server_stat_file}
while read stat_line
do
    disk_available_min_ratio=`echo ${stat_line} | awk '{print $10}'`
    disk_available_ratio=`echo ${disk_available_min_ratio} | grep -o '[0-9]\+'`
    expr ${disk_available_ratio} + 0 &>/dev/null
    if [ $? -eq 0 ]; then
        disk_used_ratio=`expr 100 - ${disk_available_ratio}`
        if [ ${disk_used_ratio} -gt ${max_disk_used_ratio} ]; then
            max_disk_used_ratio=${disk_used_ratio}
        fi
    fi

    if [ ${disk_total_MB} -eq 0 ]; then
        disk_capacity_total=`echo ${stat_line} | awk '{print $12}'`
        disk_capacity=`echo ${disk_capacity_total} | grep -o '[0-9]\+'`
        expr ${disk_capacity} + 0 &>/dev/null
        if [ $? -eq 0 ]; then
            disk_total_MB=${disk_capacity}
        fi
    fi

    memused_res=`echo ${stat_line} | awk '{print $18}'`
    memory_used_MB=`echo ${memused_res} | grep -o '[0-9]\+'`
    expr ${memory_used_MB} + 0 &>/dev/null
    if [ $? -eq 0 ]; then
        if [ ${memory_used_MB} -gt ${max_memory_used_MB} ]; then
            max_memory_used_MB=${memory_used_MB}
        fi
    fi
done<${server_stat_file}

table_size_mb=`echo app_stat | ./run.sh shell --cluster ${cluster} | grep ${app_name} | awk '{print $17}'`

echo "table ${app_name} size: ${table_size_mb}"
echo "max_memory_used_MB: ${max_memory_used_MB}"
echo "memory_total_MB: ${memory_total_MB}"
echo "max_disk_used_ratio: ${max_disk_used_ratio}"
echo "disk_total_MB: ${disk_total_MB}"

disk_radio=`echo | awk 'BEGIN{printf "%.f\n",('$table_size_mb'/'$disk_total_MB'*100)}'`
disk_used_radio_after_split=`echo | awk 'BEGIN{printf "%.0f\n",('$max_disk_used_ratio'+'$disk_radio')}'`
echo "disk_used_radio_if_split: ${disk_used_radio_after_split}"
if [ ${disk_used_radio_after_split} -gt ${disk_max_used_percent_after_split} ]; then
    echo "ERROR: partition split app ${app_name} will lead disk_used_percent exceed to ${disk_max_used_percent_after_split}%"
    exit 1
fi

memory_radio=`echo | awk 'BEGIN{printf "%.f\n",('$max_memory_used_MB'/'$memory_total_MB'*100)}'`
memory_used_radio_after_split=`echo | awk 'BEGIN{printf "%.0f\n",('$max_memory_used_percent'+'$memory_radio')}'`
echo "memory_used_radio_if_split: ${memory_used_radio_after_split}"
if [ ${memory_used_radio_after_split} -gt ${memory_max_used_percent_after_split} ]; then
    echo "ERROR: partition split app ${app_name} will lead disk_used_percent exceed to ${memory_max_used_percent_after_split}%"
    exit 1
fi

#set meta steady
echo "set_meta_level steady" | ./run.sh shell --cluster ${cluster}  &>/tmp/$UID.$PID.pegasus.set_meta_level
set_ok=`grep 'control meta level ok' /tmp/$UID.$PID.pegasus.set_meta_level | wc -l`
if [ $set_ok -ne 1 ]; then
  echo "ERROR: set meta level to steady failed"
  exit 1
fi
echo

# partition split
echo "Starting parition split..."
split_file="/tmp/$UID.$PID.pegasus.partition_split"
echo "partition_split ${app_name} ${partition_count}" | ./run.sh shell --cluster ${cluster} &>${split_file}

error_count=`grep 'failed' ${split_file} | wc -l`
if [ $error_count -ge 1 ]; then
    error=`grep 'failed' ${split_file}`
    echo "ERROR: partition split failed, error is ${error}"
    exit 1
else
    echo "partition split succeed"
fi

# wait
echo
echo "Wait for 10 minutes to wait all partition healthy..."
scount=0
while true
do
    if [ ${scount} -gt 60 ]; then
        if [ ${unwritable} -gt 0 || ${unreadable} -gt 0 ]; then
            echo
            echo "${app_name} still has ${unwritable} write unhealthy, ${unreadable} read unhealthy partition."
            echo "WARNING: please check partition split result."
            rm -f /tmp/$UID.$PID.pegasus.* &>/dev/null
            exit 0
        else
            echo
            echo "WARNING: partition split finished, but still has ${total_unhealthy} unhealthy partition, table ${app_name} still learn"
        fi
    fi

    app_detail_file="/tmp/$UID.$PID.pegasus.app.details"
    echo "app ${app_name} -d" | ./run.sh shell --cluster ${cluster} &>${app_detail_file}

    total_healthy=`grep "fully_healthy_partition_count" ${app_detail_file} | awk '{print $3}'`
    total_unhealthy=`grep "unhealthy_partition_count" ${app_detail_file} | grep -v "write" | grep -v "read" | awk '{print $3}'`
    unwritable=`grep "write_unhealthy_partition_count" ${app_detail_file} | awk '{print $3}'`
    unreadable=`grep "read_unhealthy_partition_count" ${app_detail_file} | awk '{print $3}'`

    if [ "${total_healthy}" == "${partition_count}" ]; then
        break
    else
        echo "${app_name} still has ${total_unhealthy} unhealthy, ${unwritable} write unhealthy, ${unreadable} read unhealthy partition."
    fi

    sleep 10
    scount=$((scount+1))
done

# record finish time
all_finish_time=`date +%s`
echo
echo "Finish time: `date -d @${all_finish_time} +"%Y-%m-%d %H:%M:%S"`"
echo "Partition split done, elapsed time is $((all_finish_time - all_start_time)) seconds."

#clear tmp files
rm -f /tmp/$UID.$PID.pegasus.* &>/dev/null
