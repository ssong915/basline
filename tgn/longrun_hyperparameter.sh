#! /bin/bash
EMBEDDINGS=(512)
ALPHAS=(0.2 0.4 0.6 0.8 1)
BETAS=(0.2 0.4 0.6 0.8 1)


for emb_size in ${EMBEDDINGS[*]}
do

case $emb_size in
128)
learning_rate=0.001
;;
256)
learning_rate=0.001
;;
512)
learning_rate=0.0005
;;
1024)
learning_rate=0.0005
;;
esac

for alpha in ${ALPHAS[*]}
do

for beta in ${BETAS[*]}
do

python3 -u main.py --gpu_index=0\
	--batch_size=100 \
	--eval_batch_size=64 \
	--max_sentence=20 \
	--embed_size=${emb_size} \
	--num_layer=1 \
	--num_head=4 \
	--d_hid=256 \
	--dropout=0.3 \
	--alpha=${alpha} \
	--beta=${beta} \
	--num_epochs=50 \
	--dataset=SEMEVAL \
	--learning_rate=0.0005 \
&> logs/221004-knowledge-effects-ModE/H4L1_emb${emb_size}_alpha${alpha}_beta${beta}.log

    sleep 10

done
done
done
