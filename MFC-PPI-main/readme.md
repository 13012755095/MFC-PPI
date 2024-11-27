# MFC-PPI
MFC-PPI: Protein-Protein Interaction Prediction with Multimodal Feature Fusion and Contrastive Learning
## Dependencies
MFC-PPI runs on Python 3.9.7
The required dependencies can be found in environment.yml
```
cd MFC-PPI-main
conda env create -f environment.yml
conda activate MFC-PPI
```

## Dataset
The processed data set used in this article is stored in protein_info
Due to the large size of the file, it needs to be obtained at the following link:
https://pan.baidu.com/s/1YAQG8HJVNjQqbDmGTdTveg?pwd=n2p5


## Experimental result
The experimental results of this paper are saved in the result_save folder to facilitate 
the repetition of the experiment using the model parameters.
Similarly, result_save requires the following link:
https://pan.baidu.com/s/1YAQG8HJVNjQqbDmGTdTveg?pwd=n2p5


or you can just run loop_runner.py
```
python -u /yourProject/loop_runner.py
```
or run the following script
```
python -u /yourProject/model_train.py 
--ppi_path ./protein_info/protein.actions.SHS27k.STRING.pro2.txt 
--pseq ./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv 
--split random 
--p_feat_matrix ./protein_info/protein.nodes.SHS27k.D12.pt
--p_adj_matrix ./protein_info/protein.rball.edges.SHS27kD12.npy 
--save_path ./result_save 
--epoch_num 300
```