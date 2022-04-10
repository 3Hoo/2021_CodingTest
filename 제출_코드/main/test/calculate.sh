mode=DSA

#mkdir -p ../../stats/stats_1225

#mkdir -p ../../stats/stats_1225/CNN

echo -e "................... Calculate Error OF '${mode}' ..................."

for ep in `seq 2000 -2 1000`; do
    echo -e '================ Processing epoch '${ep}' ================'
    python3 -u calculate_dsa.py --model_type=${mode} --model_dir=../../models/model_1224_dsa/model_d --epoch=${ep} --log_dir=./test_logs/json/result_${mode}.json || exit 1;
    echo -e '================ Done epoch '${ep}' ================ \n\n\n'
done
