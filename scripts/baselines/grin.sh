###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-09 21:12:49
### 
cd ../../baselines
log_path="../logs/baselines/grin"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

cuda=2

# # MAR
# dataset="METR-LA"
# miss_mechanism="MAR"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="Seattle"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Chengdu"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Shenzhen"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="PEMS-BAY"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


miss_mechanism="MNAR"
# MNAR
# dataset="METR-LA"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

# dataset="Seattle"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Chengdu"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done


# dataset="Shenzhen"

# for ((i=2021; i<=2023; i++))
# do
#   seed=$i
#   nohup python -u grin.py \
#   --dataset-name $dataset \
#   --miss_mechanism $miss_mechanism \
#   --cuda $cuda \
#   --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
#   wait
# done

dataset="PEMS-BAY"

for ((i=2021; i<=2023; i++))
do
  seed=$i
  nohup python -u grin.py \
  --dataset-name $dataset \
  --miss_mechanism $miss_mechanism \
  --cuda $cuda \
  --seed $seed > ${log_path}/${dataset}_${miss_mechanism}_$seed.log 2>&1 &
  wait
done
