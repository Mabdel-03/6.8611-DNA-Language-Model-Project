# 6.8611 Final Project: Cell Type Specific DNA Language Models

## Section 1: Vanilla Input Cell Type Specific Model Training
**Description:** Training code for our cell type specific (non architectural changes) version of the Enformer model from Žiga Avsec et al., 2021

**Found In:**/Cell_Type_Specific_Training

**Contents**:

1.  Train_Model_OG.sh: bash script submitting batch job to slurm scheduler on compute cluster to run Train_Model_OG.py
2.  Train_Model_OG.py: python file containing training code

## Section 2: Autoencoder Head Cell Type Specific Model Training
**Description:** Training code for our autoencoder headed cell type specific version of the Enformer model from Žiga Avsec et al., 2021

**Found In:**/AutoEnc-CellTypeSpec_Traiing

**Contents:** 
1.  /Autoencoder_Training: Training scripts for the autoencoder head, trained on reconstruction loss
  a. Train_Autoencoder.sh: Batch job submission for python file
  b. Train_Autoencoder_v2.py: Python file training autoencoder head on reconstruction loss
2.  /Full_Model_Training: Training scripts for the full model including reconstruction loss, trained on expression prediction
  a. Train_Full_Model.sh: Batch job submission for python file
  b. Train_Full_Model.py: Python script training full model, including pretrained autoencoder on gene expression prediction

## Section 3: Evaluation
**Found In**: /pearson_correlation_code.ipynb

**Contents**: 

Code to produce evaluation plots for trained models
