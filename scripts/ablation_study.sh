###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-10 10:14:31
### 

log_path="./logs/METR-LA/MCAR"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

# nohup python -u main.py \
#   --config_path "configs/METR-LA.yaml" \
#   --seed 0 \
#   --learnable 1 > ${log_path}/learnablepos.log 2>&1 &
# # wait

nohup python -u main.py \
  --config_path "configs/METR-LA.yaml" \
  --seed 0 \
  --learnable 0 > ${log_path}/nonlearnablepos.log 2>&1 &