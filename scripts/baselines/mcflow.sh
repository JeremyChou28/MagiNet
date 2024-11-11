###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-09 21:12:49
### 
cd ../../baselines
log_path="../logs/baselines/mcflow"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

cuda=1

miss_mechanism="MAR"
# # MAR
# dataset="METR-LA"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="Seattle"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


dataset="Chengdu"

for ((i=2021; i<=2022; i++))
do
  seed=$i
  nohup python -u mcflow.py \
  --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
  --device $cuda \
  --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
  wait
done


# dataset="Shenzhen"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="PEMS-BAY"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# miss_mechanism="MNAR"
# MNAR
# dataset="METR-LA"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="Seattle"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Chengdu"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Shenzhen"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="PEMS-BAY"

# for ((i=2021; i<=2022; i++))
# do
#   seed=$i
#   nohup python -u mcflow.py \
#   --config_path "configs/${miss_mechanism}/${dataset}.yaml" \
#   --device $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done
