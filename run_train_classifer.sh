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
    echo "Making models for ${data_name} (started at $(date)) ..."
    
    log_file=${LOG_DIR}/${THIS}__${data_name}.log
    err_file=${LOG_DIR}/${THIS}__${data_name}.err
    
    python train_classifer.py ${data_name} > ${log_file} 2> ${err_file}

    if [ $? -gt 0 ]; then
        echo "${data_name} ---- not completed -- $(date)"
    else
        echo "${data_name} ---- completed -- $(date)"
    fi
    
}
 

####################### 
# main ################

execute mnist &
pid_1=$!

execute cifar10 &
pid_2=$!

execute fmnist &
pid_3=$!

wait $pid_1 $pid_2 $pid_3
echo "ALL done -- $(date)"

