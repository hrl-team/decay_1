# decay_1
The repository contains 6 datasets two-armed bandit task (subset of Palminteri, Behavioral Neuroscience, 2023), plus the codes to fit reinforcement learning models with bias, with decay or both.  
The datasets are as follows
L2017a: Lefebvre et al, NHB, 2017 (experiment 1; partial feedback)
L2017b: Lefebvre et al, NHB, 2017 (experiment 2; partial feedback)
P2017a: Palminteri et al, PLoS CB, 2017 (experiment 1; partial feedback)
P2017b: Palminteri et al, PLoS CB, 2017 (experiment 1; complete feedback)
S2021p: Sugawara & Katahira, SciRep, 2021 (partial feedback session)
S2021c: Sugawara & Katahira, SciRep, 2021 (complete feedback session)

There are three kinds of code 
Fitting_X_Final_Decay_Anneal.m (where 'X' can be 'Partial' or 'Complete') is the code that extracts the data, launches the fitting and plots the results. 
Priors_X_Final_Decay_Anneal.m attributes priors to the parameters (based on Daw et al, Neuron, 2011)
Models_X_Final_Decay_Anneal.m calculate the likely of the models (there are three models: a biased one, one with a decay parameter on the learning rate, and one with both bias and decay). 

The codes require Matlab and the optimization toolbox. 



