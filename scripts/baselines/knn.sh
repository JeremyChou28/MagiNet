cd ../../baselines
log_path="../logs/baselines/knn"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi
# MAR
nohup python -u knn.py --config_path "configs/MAR/METR-LA.yaml" > "$log_path/METR-LA_MAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MAR/Seattle.yaml" > "$log_path/Seattle_MAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MAR/Chengdu.yaml" > "$log_path/Chengdu_MAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MAR/Shenzhen.yaml" > "$log_path/Shenzhen_MAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MAR/PEMS-BAY.yaml" > "$log_path/PEMS-BAY_MAR.log" 2>&1 &
wait

# MNAR
nohup python -u knn.py --config_path "configs/MNAR/METR-LA.yaml" > "$log_path/METR-LA_MNAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MNAR/Seattle.yaml" > "$log_path/Seattle_MNAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MNAR/Chengdu.yaml" > "$log_path/Chengdu_MNAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MNAR/Shenzhen.yaml" > "$log_path/Shenzhen_MNAR.log" 2>&1 &
wait

nohup python -u knn.py --config_path "configs/MNAR/PEMS-BAY.yaml" > "$log_path/PEMS-BAY_MNAR.log" 2>&1 &
wait