#! /bin/bash
RANDOM_SEED=(915 2020)
N_DEGREE=(10 20 30)
MESSAGE_DIM=(50 100)


# for emb_size in ${EMBEDDINGS[*]}
# do

# case $emb_size in
# 128)
# learning_rate=0.001
# ;;
# 256)
# learning_rate=0.001
# ;;
# 512)
# learning_rate=0.0005
# ;;
# 1024)
# learning_rate=0.0005
# ;;
# esac

for random_seed in ${RANDOM_SEED[*]}
do

for n_degree in ${N_DEGREE[*]}
do

for message_dim in ${MESSAGE_DIM[*]}
do

python3 -u /home/dake/workspace/HIT/Neural_Higher-order_Pattern_Prediction/baselines_nn_2.py \
	--random_seed=${random_seed} \
	--n_degree=${n_degree} \
	--message_dim=${message_dim} \
    
&> logs_hitdecoder/random_seed_${random_seed}_n_degree_${n_degree}_message_dim_${message_dim}.log

    sleep 5

done
done
done
