% this version estimates all model parameters in one go

function lik = Models_Partial_Final_Decay_Anneal(params,s,a,r,model)


% getting the number of statess 
statelabels=unique(s);

% initializing the state-specific trial counter
counter=zeros(1,max(statelabels));


if model == 1 % bias model 
    beta  = params(1);
    lr_p   = params(2); % learning rate +
    lr_n   = params(3); % learning rate +
    

elseif model==2 % declay model 
    
    beta  = params(1);
    dec   = params(3); % decay
    
    % loop over the states to create state-specific learning rates
    for x=1:max(statelabels)

        lr_d(x)=params(2);

    end



elseif model == 3 % bias + decay model
    beta  = params(1);

    dec_p   = params(4); % positive learning rate decay +
    dec_n   = params(5); % negative learning rate decay -

    % loop over the states to create state-specific learning rates

    for x=1:max(statelabels)

        lr_p(x)=params(2); % learning rate +
        lr_n(x)=params(3); % learning rate -
    end
    
end




% initializing the hidden values

Q       = zeros(max(s),2); %  Q-values

lik=0;


for i = 1:length(a)

    counter(s(i))=counter(s(i))+1; % incrementing the state counter

    if (a(i)==1 || a(i)==2) % to exclude missed reponses
% softmax
        lik = lik + log (1/(1+ exp(-beta*(Q(s(i),a(i))-Q(s(i),3-a(i))))));
        
        PEc =  r(i) - Q(s(i),a(i)); % prediction error
        
        %% bias model 
        if model==1


            Q(s(i),a(i)) = Q(s(i),a(i)) + lr_p * PEc * (PEc>0)...
                                        + lr_n * PEc * (PEc<0);

            %% decay model 
        elseif model ==2 %





            Q(s(i),a(i)) = Q(s(i),a(i)) + lr_d(s(i)) * PEc;

            lr_d(s(i))=lr_d(s(i))*dec;




            %% bias + decay model 
        elseif model ==3 %









            Q(s(i),a(i)) = Q(s(i),a(i)) + lr_p(s(i)) * PEc * (PEc>0)...
                                        + lr_n(s(i)) * PEc * (PEc<0);

            lr_p(s(i))=lr_p(s(i))*dec_p;
            lr_n(s(i))=lr_n(s(i))*dec_n;




        end

    end
end


lik = -lik;  % negative likelihood vector 

