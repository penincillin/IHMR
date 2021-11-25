## Data Preparation
- Download and unzip the data from [OneDrive Link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155102588_link_cuhk_edu_hk/EjI0fo6N-L5HpZzK9wX00Q4BlK5Ktz2r4xNuprYaMdJJiw?e=RXHlbc)
- Place the data folder (ihmr_data) in the folder of IHMR/ and organize the data folder (ihmr_data) as
```
ihmr_data/
    hand26m/
        annotation/
        image/
            train/
            test/
        param/
            train/
            test/
        prediction/
    model/
        pretrain/
        blur_kernel/
```

## Installation
### Prerequisites 
- CUDA 10.2 (not necessary, also CUDA version should also work)

### Clone the Repo
```bash
git clone git@github.com:penincillin/IHMR.git
cd IHMR
```

### Conda Environment
```bash
conda env create --name ihmr --file=docs/ihmr.yml
source activate ihmr
```

### Third Packages 
- Install [SDF package](https://github.com/penincillin/SDF_ihmr) for collision detection. 
    + Make sure the above environment is corretly set up before installing this package.

- Install utility code
```bash
git clone git@github.com:penincillin/Tools.git
cd Tools/package
python setup.py develop
```