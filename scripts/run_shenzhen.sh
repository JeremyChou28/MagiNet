###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-09 15:55:18
### 

# MAR
log_path="./logs/Shenzhen/MNAR"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

for ((i=2021; i<=2023; i++))
do
  seed=$i
  nohup python -u main.py \
  --config_path "configs/MNAR/Shenzhen.yaml" \
  --seed $seed > ${log_path}/$seed.log 2>&1 &
  wait
done
