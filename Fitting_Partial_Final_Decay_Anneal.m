% This code finds the optimal parameters
% This code works for the partial feedback experiments
% This code requires Matlab Optimization toolbox

clear all
close all

%% Choosing the dataset

 load L2017a % Lefebvre 2017 (fMRI experiment n=50)
% load L2017b % Lefebvre 2017 (Lab experiment n=35)
% load P2017a % Palminteri 2017 (Partial feedback experiment n=20)
% load S2021p;sta=stap;cho=chop;out=outp; % Sugawara & Katahira 2021 (partial sessions)


%% what is in the data files

% sta: state or pair of symbols (unique identifier).
% cho: choices 1 or 2 (in the option space, not the motor space)
% out: obtained outcome (rescaled, across experiments: -1 or 1)

subjecttot=numel(sta); % this get ther number of subjects, by analysing how many elements there are in one of the variables

%%


n_model=3; % because we are fitting two models

nfpm=[3 3 5]; % free parameters per model: bias=3, decay=3, bias+decay=5

% fmincon settings (modified from default to include more iterations and function evaluations)

options = optimset('Algorithm', 'interior-point', 'Display', 'off', 'MaxIter', 10000,'MaxFunEval',10000);

nsub=0;

% subject loop
for k_sub = 1:subjecttot

    % model loop
    for k_model = 1:n_model

        % prepare starting points and parameter bounds
        if     k_model < 3  % bias & decay
            lb = [0 0 0];        LB = [0 0 0];   % lower bounds xb=to generated starting points / XB=true
            ub = [15 1 1];       UB = [Inf 1 1]; % upper bounds xb=to generated starting points / XB=true
        elseif k_model == 3  % full model: bias + decay
            lb = [0 0 0 0 0];       LB = [0 0 0 0 0]; % lower bounds xb=to generated starting points / XB=true
            ub = [15 1 1 1 1];       UB = [Inf 1 1 1 1]; % upper bounds xb=to generated starting points / XB=true
        end

        ddb = ub - lb; % where to look for the random point initialization

        % prepare multiple starting points for estimation
        n_rep           = 5;
        parameters_rep  = NaN(n_rep,nfpm(k_model));     parametersLPP_rep  = NaN(n_rep,nfpm(k_model));
        ll_rep          = NaN(n_rep,1);                 LPP_rep            = NaN(n_rep,1);
        FminHess        = NaN(n_rep,nfpm(k_model),nfpm(k_model));

        for k_rep = 1:n_rep
            % prepare starting points and parameter bounds
            x0 = lb + rand(1,length(lb)).*ddb;
            x0 = x0(1:nfpm(k_model));


            % run ML and MAP estimations
            [parametersLPP_rep(k_rep,1:nfpm(k_model)),LPP_rep(k_rep),~,~,~,~,FminHess(k_rep,:,:)]=fmincon(@(x) Priors_Partial_Final_Decay_Anneal(x,sta{k_sub},cho{k_sub},out{k_sub},k_model),x0,[],[],[],[],LB,UB,[],options);
        end

        % find best params over repetitions & store optimization outputs

        [~,posLPP]                                      = min(LPP_rep);
        parametersLPP(k_sub,k_model,1:nfpm(k_model))    = parametersLPP_rep(posLPP(1),1:nfpm(k_model));
        LPP(k_sub,k_model)                              = LPP_rep(posLPP(1),:) - nfpm(k_model)*log(2*pi)/2 + real(log(det(squeeze(FminHess(posLPP(1),:,:)))))/2;

        check_conv(k_sub)                               =  ~any(eig(squeeze(FminHess(posLPP(1),:,:)))<0);


    end
end


%% Saving the outputs 
% uncomment if relevant 
% save NAME LPP parametersLPP


%% Plotting start here

%% model comparison plot
deltaLPP(:,1)=LPP(:,2)-LPP(:,1);
deltaLPP(:,2)=LPP(:,3)-LPP(:,1);
figure
scatter(deltaLPP(:,1),deltaLPP(:,2),...
    'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0.5 0.5 0.5])
xlim([-max(abs(deltaLPP(:,1))) max(abs(deltaLPP(:,1)))])
ylim([-max(abs(deltaLPP(:,2))) max(abs(deltaLPP(:,2)))])

grid('on')

title('Model comparison')
xlabel('"Bias" better than "decay"')
ylabel('"Bias" better than "full"')
set(gca,'FontSize',16)

%% parameters plot
figure
subplot(1,2,1)
LRCONF(:,:)=parametersLPP(:,1,2:3);
for n=1:length(LRCONF)
    scatter([1 2],LRCONF(n,:),'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.8 .8 .8])
    hold on
end

plot(LRCONF','Color',[.8 .8 .8])
hold on
errorbar([1 2],mean(LRCONF),sem(LRCONF),'Linewidth',2,'Color',[.1 .5 .9])
hold on
scatter([1 2],mean(LRCONF),'MarkerFaceColor',[.1 .5 .9],'MarkerEdgeColor',[.1 .5 .9],'Linewidth',5)
axis([0.8 2.2  0 1])
ylabel('Learning rate')
set(gca,'FontSize',16,...
    'XTickLabel',{'\alpha_+','\alpha_-'},...
    'YTick',[0 .2 .4 .6 .8 1] ,...
    'XTick',[1:2]);

title('Bias model','Fontsize',16)

subplot(1,2,2)
LRFULL(:,:)=parametersLPP(:,2,2:3);
for n=1:length(LRFULL)
    scatter([1 2],LRFULL(n,:),'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.8 .8 .8])
    hold on
end

plot(LRFULL','Color',[.8 .8 .8])
hold on
errorbar([1 2],mean(LRFULL),sem(LRFULL),'Linewidth',2,'Color',[.9 .5 .9])
hold on
scatter([1 2],mean(LRFULL),'MarkerFaceColor',[.9 .5 .9],'MarkerEdgeColor',[.9 .5 .9],'Linewidth',5)
axis([0.8 2.2  0 1])
ylabel('Learning rate')
set(gca,'FontSize',16,...
    'XTickLabel',{'\alpha','\delta'},...
    'YTick',[0 .2 .4 .6 .8 1] ,...
    'XTick',[1:2]);
 title('Decay model','Fontsize',16)
% suptitle('L2017a','Fontsize',16)
box off

%% plotting the full model 
figure
subplot(1,2,1)
LRCONF(:,:)=parametersLPP(:,3,2:3)
for n=1:length(LRCONF)
    scatter([1 2],LRCONF(n,:),'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.8 .8 .8])
    hold on
end

plot(LRCONF','Color',[.8 .8 .8])
hold on
errorbar([1 2],mean(LRCONF),sem(LRCONF),'Linewidth',2,'Color',[.1 .5 .9])
hold on
scatter([1 2],mean(LRCONF),'MarkerFaceColor',[.1 .5 .9],'MarkerEdgeColor',[.1 .5 .9],'Linewidth',5)
axis([0.8 2.2  0 1])
ylabel('Learning rate')
set(gca,'FontSize',16,...
    'XTickLabel',{'\alpha_+','\alpha_-'},...
    'YTick',[0 .2 .4 .6 .8 1] ,...
    'XTick',[1:2]);

title('Learning rates','Fontsize',16)

subplot(1,2,2)
LRFULL(:,:)=parametersLPP(:,3,4:5);
for n=1:length(LRFULL)
    scatter([1 2],LRFULL(n,:),'MarkerFaceColor',[.8 .8 .8],'MarkerEdgeColor',[.8 .8 .8])
    hold on
end

plot(LRFULL','Color',[.8 .8 .8])
hold on
errorbar([1 2],mean(LRFULL),sem(LRFULL),'Linewidth',2,'Color',[.9 .5 .9])
hold on
scatter([1 2],mean(LRFULL),'MarkerFaceColor',[.9 .5 .9],'MarkerEdgeColor',[.9 .5 .9],'Linewidth',5)
axis([0.8 2.2  0 1])
ylabel('Learning rate')
set(gca,'FontSize',16,...
    'XTickLabel',{'\delta_+','\delta_-'},...
    'YTick',[0 .2 .4 .6 .8 1] ,...
    'XTick',[1:2]);
 title('Decays','Fontsize',16)
% suptitle('L2017a','Fontsize',16)
box off




%% generating the trial-by-trial learning rates of the bias+decay model 
LRs(:,1:2)=parametersLPP(:,3,2:3);%isolating the learnig rates of the b+d model
DEs(:,1:2)=parametersLPP(:,3,4:5);%isolating the decays of the b+d model

for k_sub = 1:subjecttot % subject loop
    
    lr_p_curve(k_sub,1)=LRs(k_sub,1);% the first trial learning rate +
    lr_n_curve(k_sub,1)=LRs(k_sub,2);% the first trial learning rate - 
    
    
    for trial=2:23 % trial loop
        lr_p_curve(k_sub,trial)=lr_p_curve(k_sub,trial-1)*DEs(k_sub,1);% the n trial learning rate +
        lr_n_curve(k_sub,trial)=lr_n_curve(k_sub,trial-1)*DEs(k_sub,2);% the n trial learning rate -
    end
    

end


%% plotting the trial-by-trial learning rates of the b+d model
figure
subplot(1,2,1)

for k_sub = 1:subjecttot % individual curves

    plot(lr_p_curve(k_sub,:),'Color',[.35 .75 .95],'LineWidth',1)
    hold on

end
plot(mean(lr_p_curve),'Color',[.1 .5 .9],'LineWidth',4) % average
title('\alpha_+','Fontsize',16)
xlabel('Trial','Fontsize',16)
set(gca,'FontSize',16)

subplot(1,2,2)

for k_sub = 1:subjecttot % individual curves
    
    plot(lr_n_curve(k_sub,:),'Color',[.95 .75 .95],'LineWidth',1) % average
    hold on

end

plot(mean(lr_n_curve),'Color',[.9 .5 .9],'LineWidth',4)
title('\alpha_-','Fontsize',16)
xlabel('Trial','Fontsize',16)
set(gca,'FontSize',16)
%%

