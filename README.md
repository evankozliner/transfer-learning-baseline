# transfer-learning-baseline

# from the image_retrain dir
```bash
python tensorflow/tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/image_retrain/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/image_retrain/inception \
--output_graph=/image_retrain/retrained_graph.pb \
--output_labels=/image_retrain/retrained_labels.txt \
--image_dir /64x64_blocks
```


# For using docker
```bash
docker run -it -v $HOME/Documents/research/image-retrain-with-validation/64x64_blocks:/64x64_blocks gcr.io/tensorflow/tensorflow:latest-devel
```


