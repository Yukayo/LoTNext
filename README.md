# Taming the Long Tail in Human Mobility Prediction
![image](./data/LoTNext.png)
# Environments
```
conda create -n LoTNext python==3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
```

# Data Preparation

**Note: For steps 1 and 2, choose one to execute.**

**1. place [check-in_data.zip](https://drive.google.com/file/d/1WzwP6NFZ3rvSFLy8rXzhuqzM_MgNJFmF/view?usp=sharing) into ./data/ and unzip the file as following:**

./data/checkins-gowalla.txt

./data/checkins-4sq.txt

Note: These are raw check-in datasets, if you choose this way you need to comment out the code of train.py
```
with open(setting.loader_file, 'rb') as f:
    poi_loader = pickle.load(f)
```

**2. place [check-in_dataloader.zip](https://drive.google.com/file/d/12yvZ7ClT3klDNI2cqIYyBhpAFV6pQntC/view?usp=sharing) into ./data/ and unzip the file as following:**

./data/poi_loader-gowalla.pkl

./data/poi_loader-4sq.pkl

Note: Unzip, you don't need to comment out any codes.

**3. place [Graphs.zip](https://drive.google.com/file/d/1nj916wbuRvSLKPB8ddqktmpVYbkSIa4y/view?usp=sharing) into ./KGE/Graphs/ and unzip the file as following :**

./KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl


# Model Training

Gowalla
```
python train.py --dataset checkins-gowalla.txt --dataloader poi_loader-gowalla.pkl --trans_loc_file ./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl --trans_interact_file ./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl --log_file ./results/log_gowalla --model_file ./model_log/model_gowalla

```
Foursquare
```
python train.py --dataset checkins-4sq.txt --dataloader poi_loader-4sq.pkl --trans_loc_file ./KGE/POI_graph/foursquare_scheme2_transe_loc_temporal_20.pkl --trans_interact_file ./KGE/POI_graph/foursquare_scheme2_transe_user-loc_100.pkl --log_file ./results/log_foursquare --model_file ./model_log/model_4sq

```
