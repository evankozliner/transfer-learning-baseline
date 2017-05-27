# How to Get Transfer Learning Results

1. **Start instance**


```bash
  cd aws_helpers
  python aws_tool.py start
```

This will spit out an ip, copy that IP. If you don't copy it correctly you can always run: ` python aw_tool.py ip ` to get the IP.


2. **Crop the images locally**


You will need to edit the cropping file with the appropriate block size
This will create a directory name data_blocksizexblock_size

``` 
python image_cropper 
```

3. **Copy the images over to the remote server**

I will give you the key to copy to the instance 

Zip the images

` zip -r imgs.zip folder-created-by-crop-script`

` scp -i ~/your-key-path/key-i-gave-you.pem zipped-folder  ubuntu@ip-you-copied:~`

4. **SSH into the instance & unzip images**

` sudo ssh -i ~/your-key-path/key-i-gave-you.pem ubuntu@ip-you-copied `

``

Move the images to the git directory
`mv zipped-images.zip transfer-learning-baseline`
`unzip zipped-images.zip`

4. **Start tmux**

```bash 
tmux 
```

5. **Run Docker**

```bash 
docker run -it -v $HOME/directory-with-images-in-classes/:/data gcr.io/tensorflow/tensorflow:latest-devel 

# In docker
cd /tensorflow


python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/inception \
--output_graph=/retrained_graph.pb \
--output_labels=/retrained_labels.txt \
--image_dir /data \
--summaries_dir /summaries \
--testing_percentage 5 \
--validation_percentage 15 \
--print_misclassified_test_images \
--random_crop 5 \
--random_scale 5 \
--random_brightness 5 \
--flip_left_right
```

At this point you can exit tmux with ctrl+b+d, then exit the server. Come back in a number of hours and it should be ready...

6. **Get the test accuracy**

First you will need to reattach to tmux and be sure it is done training:

``` tmux attach ```

Rename the graph some appriopriate: 

``` mv retrained_graph.pb better-name.pb ``` e.g. retrained_graph_64x64.pb for the 64x64 block graph.

Copy the learned graph to the server 

``` sudo docker cp $(docker ps -alq):/path-to-retrained_graph.pb ```

Edit `whole_img_classifier.py` with the name of the retrained graph (called GRAPH_FILENAME)

Then run the test script: ``` python whole_img_classifier ```

Then exit tmux again with ctrl+b+d, exit the server, wait a couple hours and come back to the server once more.

The script should have output the test accuracy, record it and continue.

7. Copy the trained graphs locally

``` scp -i ~/your-key-path/key-i-gave-you.pem ubuntu@ip-you-copied:~/path-to-graph-you-named .```


