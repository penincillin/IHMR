## Training

### Prepare Real-time Visualization
To visualize the training results and the loss curve in real-time, please run ```python -m visdom.server 8201``` and click the URL [http://localhost:8201](http://localhost:8201).
The port number can be changed by modifying the ```visdom_port``` in [bash/train_baseline.sh](bash/train_baseline.sh).

### Train IHMR-Baseline
```
sh bash/train_baseline.sh
```
The train logs will be stored in ```log/baseline/train_log/[date-time].log```. The intermediate results will be stored in ```checkpoints/baseline/```


### Train IHMR-MLP
```
sh bash/train_mlp.sh
```
The train logs will be stored in ```log/mlp/train_log/[date-time].log```. The intermediate results will be stored in ```checkpoints/mlp/``