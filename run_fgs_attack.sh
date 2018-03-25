#!/bin/bash


# redirect standard in/error
LOG_DIR="$(dirname ${0})/logs"
mkdir -p $LOG_DIR

THIS=$(basename ${0} .sh)

exec > ${LOG_DIR}/${THIS}.log
exec 2> ${LOG_DIR}/${THIS}.err


# functon to execute python script
function execute() {
    data_name=$1
    model_type=$2
    echo "Making adversaries of ${data_name} against ${model_type} (started at $(date)) ..."
    
    log_file=${LOG_DIR}/${THIS}__${data_name}_${model_type}.log
    err_file=${LOG_DIR}/${THIS}__${data_name}_${model_type}.err
    
    python fgs_attack.py ${data_name} ${model_type} > ${log_file} 2> ${err_file}

    if [ $? -gt 0 ]; then
        echo "${data_name} & ${model_type} ---- not completed -- $(date)"
    else
        echo "${data_name} & ${model_type} ---- completed -- $(date)"
    fi
    
}
 

####################### 
# main ################

execute mnist base &
pid_1=$!

execute mnist distilled &
pid_2=$!


execute cifar10 base &
pid_3=$!

execute cifar10 distilled &
pid_4=$!


execute fmnist base &
pid_5=$!

execute fmnist distilled &
pid_6=$!


wait $pid_1 $pid_2 $pid_3 $pid_4 $pid_5 $pid_6
echo "ALL done -- $(date)"

