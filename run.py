import os

for i in range(10):
    os.system(
        "python model_train.py --ppi_path ./protein_info/protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv --split random --p_feat_matrix ./protein_info/x_list.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save --epoch_num 500")
