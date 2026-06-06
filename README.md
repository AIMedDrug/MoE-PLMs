# 					MoE-PLMs



Modulating protein thermal stability through mutations is widely applied in protein design for synthetic biology, thus necessitating reliable methods to predict changes in protein thermal stability upon various mutations. A prediction framework that utilizes the Sparse Mixture Expert (MoE) mechanism to adaptively fuse different embedding representations from different PLMs to predict changes in protein thermalstability upon mutations.

## Data

| Dataset | Type |                 Composition                  |                             url                              |
| ------- | ---- | :------------------------------------------: | :----------------------------------------------------------: |
| S8754   | ΔΔG  | 8754 single-point mutation from 301 proteins | [S8754](https://github.com/Gonglab-THU/GeoStab/blob/main/data/ddG/S8754.csv) |
| S669    | ΔΔG  |  669 single-point mutation from 94 proteins  | [S669](https://github.com/Gonglab-THU/GeoStab/blob/main/data/ddG/S669.csv) |
| S4346   | ΔTm  | 4346 single-point mutation from 349 proteins | [S4346](https://github.com/Gonglab-THU/GeoStab/blob/main/data/dTm/S4346.csv) |
| S571    | ΔTm  |  571 single-point mutation from 37 proteins  | [S571](https://github.com/Gonglab-THU/GeoStab/blob/main/data/dTm/S571.csv) |



## Installation

To get started with MoE-PLMs, you should have PyTorch and [conda](https://www.anaconda.com/) installed to use this repository. You can follow this command for installing the conda environment:

```
conda env create --name MoE-PLMs --file=environment.yml
```



## Usage

ALL pretrained  models can be loaded through  huggingface library.

ESM2:    [ESM2](https://huggingface.co/facebook/esm2_t6_8M_UR50D)

ESMC:    [ESMC](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12)

ProGen2:  [progen2](https://huggingface.co/hugohrban/progen2-medium)

AMPLIFY:  [AMPLIFY](https://huggingface.co/chandar-lab/AMPLIFY_350M)	

ProtT5:   [prot_t5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)



### 1.Data processing

We use script files to extract the original sequence from the dataset. Mutation sequence and predicted target are used to obtain a fasta file for subsequent operations.

```
python scripts/fasta_utils.py
```



### 2.Extract embeddings:

For each type of protein model, we have provided corresponding script files that can be used. I will introduce the usage of each script so that you can embed it into your own dataset.

```python
#example
python scripts/extract_T5.py -i #you fasta file path -o #you output path
```



### 3. Compress embeddings:(only ESM2 need) 

```python
python scripts/compressing_embeddings.py  -e embed_dir  -o out_dir -c mean -l 30
```



### 4.Mutant Site Representation Difference Extraction (Mut-Wt)

Compute a difference vector between the wild-type and mutant embeddings  

```python
#pt_del
python scripts/emb_diff/pt_subtractor.py   file1_mut.pt  file2_wt.pt  outputfile_result.pt

#pkl_del
python scripts/emb_diff/pkl_array_subtractor.py file1_mut.pkl file2_wt.pkl  outputfile_result.pkl
```



### 5.MoE-PLMs 

Using multiple obtained results for multi model fusion prediction

```python
python MoE_pred/main_predict.py  --train_x1  ProstT5_S8754_file
							  --train_x2  ESMC_600m_S8754_file
							  --train_x3  ESM2_650m_S8754_file  
							  --train_meta S8754.csv  
							  --test_x1   ProstT5_S669_file  
							  --test_x2   ESMC_600m_S669_file  
							  --test_x3   ESM2_650m_S669_file  
							  --test_meta S669.csv  
							  -o you output path
```



### 6.Transfer learning

```python
#first step, We use this script to obtain weights for a good result.
python MoE_pred/Transfer/train_save_weights.py  file_path  output_path ...

#second step, like the training process above, only requires multiple inputs of the weight path obtained above for the input part
python MoE_pred/Transfer/DTm_transfer_predict.py train_file_path test_file_path pretrained final_best_model.pth  output_path ....
```

All code paths need to be changed to paths in your own folder.













