## Testing

### Testing IHMR-Baseline
To evaluate the provided pretrain weights (```hmr_data/models/pretrain/baseline.pth```), please 
- Modify the variable ```test_epoch``` defined in ```bash/test_baseline.sh``` to ```pretrain```
- Run following commands:
```bash
mkdir -p checkpoints/baseline; cd checkpoints/baseline; 
ln -fs ../../ihmr_data/models/pretrain/baseline.pth pretrain_net_baseline.pth; cd ../../;
sh bash/test_baseline.sh
```

To evaluate on trained weights, for example the weights stored in epoch 20, please
- Modify the variable ```test_epoch``` defined in ```bash/test_baseline.sh``` to ```20```
- Run following commands:
```bash
sh bash/test_baseline.sh
```

The test logs will be stored in ```log/baseline/test_log/hand26m.log```. The evaluated results will be stored in ```evaluate_results/baseline/hand26m.pkl```.


### Testing IHMR-MLP
To evaluate the provided pretrain weights (```hmr_data/models/pretrain/mlp/```), please 
- Modify the variable ```test_epoch``` defined in ```bash/test_mlp.sh``` to ```pretrain```
- Run following commands:
```bash
mkdir -p checkpoints/mlp; cp -r ./ihmr_data/models/pretrain/mlp/*.pth checkpoints/mlp;
sh bash/test_mlp.sh
```
To evaluate on trained weights, please
- Modify the variable ```test_epoch``` defined in ```bash/test_mlp.sh``` to ```latest```
- Run following commands:
```bash
sh bash/test_mlp.sh
```
The test logs will be stored in ```log/mlp/test_log/hand26m.log```.


### Run IHMR-OPT
To run IHMR-OPT, please
```bash
sh bash/optimize.sh
```
The logs will be stored in ```log/optimize/test_log/opt.log```.


### Visualize the Predictions
To visualize the predictions and save the predicted meshes into obj files, please run
```bash
sh bash/visualize.sh
```
To visualize predictions from different datasets, please change the variable ```method``` to one of ```[baseline, mlp, opt]```.
