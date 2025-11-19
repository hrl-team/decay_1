% this version estimates all model parameters in one go

function lik = Models_Complete_Final_Decay_Anneal(params,s,a,r,c,model)

statelabels=unique(s);

counter=zeros(1,max(statelabels));

    beta  = params(1);

if model == 1 % bias model 

    lr_p   = params(2); 
    lr_n   = params(3); 
    
elseif model==2 % decay
    
    dec   = params(3); 
    
    for x=1:max(statelabels)
        
        lr_d(x)=params(2);
        
    end
    
    
    
elseif model == 3 % full model
    
    
    for x=1:max(statelabels)
        
        lr_p(x)=params(2);
        lr_n(x)=params(3);
    end
    
    
    dec_p   = params(4);
    dec_n   = params(5);
    
end




Q       = zeros(max(s),2); %  Q-values

lik=0;


for i = 1:length(a)
    
    counter(s(i))=counter(s(i))+1;
    
    if (a(i)==1 || a(i)==2) % to exclude missed reponses
        
        lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i))))));
        PEc =  r(i) - Q(s(i),a(i));
        PEu =  c(i) - Q(s(i),3-a(i));
        %% bias model 
        if model == 1
            
            
            
            
            
            Q(s(i),a(i)) = Q(s(i),a(i)) + lr_p * PEc * (PEc>0)...
                                        + lr_n * PEc * (PEc<0);
            
            Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr_n * PEu * (PEu>0)...
                                            + lr_p * PEu * (PEu<0);
            
            %% decay model
        elseif model == 2 %
            
            
            
            
            Q(s(i),a(i)) = Q(s(i),a(i))     + lr_d(s(i)) * PEc;
            
            Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr_d(s(i)) * PEu;
            
            lr_d(s(i))=lr_d(s(i))*dec;
            
            
            
            %% full model
        elseif model == 3 %
            
            
            
            
            Q(s(i),a(i)) = Q(s(i),a(i)) + lr_p(s(i)) * PEc * (PEc>0)...
                                        + lr_n(s(i)) * PEc * (PEc<0);
            
            Q(s(i),3-a(i)) = Q(s(i),3-a(i)) + lr_n(s(i)) * PEu * (PEu>0)...
                                            + lr_p(s(i)) * PEu * (PEu<0);
       
            lr_p(s(i))=lr_p(s(i))*dec_p;
            lr_n(s(i))=lr_n(s(i))*dec_n;
            
            
            
            
            
            
        end
    end
    
end


lik = -lik;                                                                % LL vector taking into account both the likelihood

