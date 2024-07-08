#! /bin/bash
RANDOM_SEED=(0)
DATA=('threads-ask-ubuntu')
freq_size=(1210000000)


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

python3 -u train_self_supervised.py \
	--n_runs=10 \
	--random_seed=${random_seed} \
	--n_degree=${n_degree} \
	--message_dim=${message_dim} \

&> logs_model/random_seed_${random_seed}_n_degree_${n_degree}_message_dim_${message_dim}.log

    sleep 5

done
done
done
