#python3 -u 1222_train.py --model_type=LSTM_Attention --base_lr=0.02 --c_lr=0.05 --bUseAngleV=False --epochs=4000 --model_save_path=../models/model_1223 --seed=1

#python3 -u 1224_train.py --model_type=DSA --base_lr=0.02 --c_lr=0.05 --bUseAngleV=False --epochs=4000 --model_save_path=../models/model_1224_dsa --seed=1

#python3 -u 1224v2_train_img.py --model_type=CNN --base_lr=0.02 --c_lr=0.05 --bUseAngleV=False --epochs=2000 --model_save_path=../models/model_1224_img --seed=1

#python3 -u 1226_train_tf.py --model_type=Transformer --base_lr=0.02 --c_lr=0.05 --bUseAngleV=False --epochs=4000 --model_save_path=../models/model_1226_tf --seed=1

python3 -u 1226_train_tf.py --model_type=TransformerCNN --base_lr=0.02 --c_lr=0.05 --bUseAngleV=False --epochs=4000 --model_save_path=../models/model_1226_tfCnn --seed=1