python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 42 --train_batch_size=16 --experiment_name=baseline
python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 42 --train_batch_size=48 --experiment_name="batchx3"
python model.py --learning_rate 2e-5 --accumulate_grad_batches 3 --seed 42 --train_batch_size=16 --experiment_name="gradx3"
python model.py --learning_rate 6e-5 --accumulate_grad_batches 3 --seed 42 --train_batch_size=16 --experiment_name="gradx3_lrx3"

python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 52 --train_batch_size=16 --experiment_name=baseline
python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 52 --train_batch_size=48 --experiment_name="batchx3"
python model.py --learning_rate 2e-5 --accumulate_grad_batches 3 --seed 52 --train_batch_size=16 --experiment_name="gradx3"
python model.py --learning_rate 6e-5 --accumulate_grad_batches 3 --seed 52 --train_batch_size=16 --experiment_name="gradx3_lrx3"

python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 62 --train_batch_size=16 --experiment_name=baseline
python model.py --learning_rate 2e-5 --accumulate_grad_batches 1 --seed 62 --train_batch_size=48 --experiment_name="batchx3"
python model.py --learning_rate 2e-5 --accumulate_grad_batches 3 --seed 62 --train_batch_size=16 --experiment_name="gradx3"
python model.py --learning_rate 6e-5 --accumulate_grad_batches 3 --seed 62 --train_batch_size=16 --experiment_name="gradx3_lrx3"