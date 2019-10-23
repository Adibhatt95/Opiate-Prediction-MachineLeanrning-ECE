# Opioid Prescription Predictive Analysis  
## Submission By  
<ul>
  <li>Aditya Bhatt (apb462@nyu.edu)</li>
</ul>
<br/>

## Source Code 
-PLEASE READ BOTH JUPYTER NOTEBOOKS, THERE ARE 2 NOTEBOOKS PLEASE READ THEM BOTH:-
- Complete Source Code along with hyperparameter tuning: `apb462_Project_Notebook_Final.ipynb` - NOTEBOOK 1
- Complete 2-layer NN code trainied with GPU with one-hot encoding: `2LayerNeuralNetwork82percentAccuracy.ipynb` - NOTEBOOK 2
- Hyperparameter tuning script: `hpc_tuning.py`
- Batch script to run in HPC: `hpc_tuning.sbatch`


## ML for Social Good

<b style="color:red">Problem:</b> Opioid Prescription helps to treat moderate to severe pain but it also leads to addiction
and hence, people misuse it by consuming it at higher rate. Overdose of Opioids leads to death.
1. How to improve ways the opioids are prescribed?
2. How to reduce the death toll due to drug overdose in North America?

<b style="color:green">Goal:</b> Detect opium components in the data and predict prescribers with opioid prescriptions
which may lead to drug addiction.

<b style="color:blue">Data:</b> Prescription data with drug components, medicare domain, opium drug list, state-wise
overdose deaths.

<b style="color:orange">Target Variable:</b> Opioid Prescriber or not?

<b style="color:purple">Impact:</b> Model is able to predict prescriber with opioid prescription and
individual chemicals and features of high importance causing opioid addiction. This can greatly reduce deaths due to drug addictions and can also improve awareness of the causes of drug addictions and can facilitate better prescriptions techniques during drug presriptions. The PCA analysis reveals the most about the most interesting findings through PCA to exemplify the above. Please see the jupyter notebooks for more information on each notebook
