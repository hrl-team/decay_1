# decay_1
The repository contains 6 datasets two-armed bandit task (subset of Palminteri, Behavioral Neuroscience, 2023), plus the codes to fit reinforcement learning models with bias, with decay or both.  
The datasets are as follows
L2017a: Lefebvre et al, NHB, 2017 (experiment 1; partial feedback)
L2017b: Lefebvre et al, NHB, 2017 (experiment 2; partial feedback)
P2017a: Palminteri et al, PLoS CB, 2017 (experiment 1; partial feedback)
P2017b: Palminteri et al, PLoS CB, 2017 (experiment 1; complete feedback)
S2021p: Sugawara & Katahira, SciRep, 2021 (partial feedback session)
S2021c: Sugawara & Katahira, SciRep, 2021 (complete feedback session)

Some of the codes have been written by Stefano Palminteri 
There are three kinds of code 
Fitting_X_Final_Decay_Anneal.m (where 'X' can be 'Partial' or 'Complete') is the code that extracts the data, launches the fitting and plots the results. 
Priors_X_Final_Decay_Anneal.m attributes priors to the parameters (based on Daw et al, Neuron, 2011)
Models_X_Final_Decay_Anneal.m calculate the likely of the models (there are three models: a biased one, one with a decay parameter on the learning rate, and one with both bias and decay). 

The other codes (the one used to generate the figure of Cecchi and Palminteri, "Genuine Learning Biases Persist After Accounting for Temporally Decreasing Learning Rates: insight from fitting six datasets") have been written by Romane Cecchi. They are extensively commented within. 

model_fitting_decay.m (fit the models to the datasets)
def_models_param.m (define the parameters priors; function called by model_fitting_decay.m)
function_model_simulations_mult_options_decay.m (define the models; function called by model_fitting_decay.m)

plot_fig_decay.m (plot the results of the fitting)

The codes require Matlab and the optimization toolbox. 



