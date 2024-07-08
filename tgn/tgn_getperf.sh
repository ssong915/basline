#! /bin/bash
RANDOM_SEED=(0 42 516 915 2020)
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

python3 -u get_embedding.py \
	--n_runs=10 \
	--n_degree=${n_degree} \
	--message_dim=${message_dim} \
&> logs_embedding/n_degree_${n_degree}_message_dim_${message_dim}.log

    sleep 5

done
done
done
