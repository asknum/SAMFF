# SAMFF
Step 1: Create a virtual environment named SAMFF and activate it.
conda create -n SAMFF python=3.8
conda activate SAMFF
Step 2: Install dependencies.
pip install -r requirements.txt

All configuration for model training and testing are stored in the local folder config
Example of Training on WHU-CD Dataset
python train.py --config/gz.json 
Example of Testing on WHU-CD Dataset
python test.py --config/gz_test_samff.json
