%% this function calculate the proability of the sampled parameters
function [post]=Priors_Partial_Final_Decay_Anneal(params,s,a,r,model)


    
    % bias and decay models 
if model < 3
    % for each parameter getting the log of the probability 
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    p = [pbeta plr1 plr2];

    %  bias + decay model 
elseif model == 3
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0)); 
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    lr3   = params(4); plr3  = log(betapdf(lr3,1.1,1.1));
    lr4   = params(5); plr4  = log(betapdf(lr4,1.1,1.1));
    p = [pbeta plr1 plr2 plr3 plr4];
    
    
end

% negative log of the proability of the parameters 
p = -sum(p); 

% calleing the model function 
l=Models_Partial_Final_Decay_Anneal(params,s,a,r,model);


post = p + l;

  