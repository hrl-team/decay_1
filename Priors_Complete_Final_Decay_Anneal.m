
function [post]=Priors_Complete_Final_Decay_Anneal(params,s,a,r,c,model)



% bias model and declay model 
if model < 3
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0));
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    p = [pbeta plr1 plr2];
    % full model
elseif model == 3
    beta  = params(1); pbeta = log(gampdf(beta,1.2,5.0));
    lr1   = params(2); plr1  = log(betapdf(lr1,1.1,1.1));
    lr2   = params(3); plr2  = log(betapdf(lr2,1.1,1.1));
    lr3   = params(4); plr3  = log(betapdf(lr3,1.1,1.1));
    lr4   = params(5); plr4  = log(betapdf(lr4,1.1,1.1));
    p = [pbeta plr1 plr2 plr3 plr4];
    
    
end

p = -sum(p);

l=Models_Complete_Final_Decay_Anneal(params,s,a,r,c,model);

post = p + l;

