%% Fitting for a three-armed bandit
% ---------------------------------
% Romane Cecchi, 2025
%
% TASK 4 options:
% 2x2 design, 45 trials per condition
% cond 1 = Wide 2 opt (W2o)
% cond 2 = Wide 4 opt (W4o)
% cond 3 = Narrow 2 opt (N2o)
% cond 4 = Narrow 4 (N4o)

clearvars
close all hidden
clear all

%% Datasets

% L = Lefebvre
% P = Palminteri
% S = Sugawara & Katahira

% L2017a, L2017b, P2017a, S2021p = partiel
% P2017b, S2021c = complete

%% Options

% MODELS: (no normalization)
% 1: 1 alpha (α± = α)
% 2: 2 alpha (α_c+ = α_u- and α_c- = α_u+)
% 3: 4 alpha (chosen/unchosen x +/-) -> Only for complete tasks
% 4: Bayes-greedy (α = deterministic = 1/(t+3))
% 5: Simple decay (1 alpha + 1 delta)
% 6: Full decay (2 alpha [as in model 2] + 2 delta [+/-])

opt.manip = 'S2021c'; % L2017a | L2017b | P2017a | P2017b | S2021c | S2021p
opt.whichmodel = 1:6; % 1:6; [1:2, 4:6]

opt.phase_fit = 'learning'; % 'learning'
opt.modeling_step = 'fitting';
opt.estimation_type = 'MLE'; % MLE (maximum‐likelihood) | MAP (maximum a posteriori)

opt.n_rep = 30; % Number of random point initialization

%% Get free parameters names and number for all models

params = def_models_param(1); % Only to retrieve param names
opt.param_name = params.names;
opt.param_nb = numel(opt.param_name);

%% Import database tables

init.path = '/Users/romane/Documents/BDD_Decay_alpha/Data'; % Path to database .csv files
db = open(fullfile(init.path, sprintf('%s.mat', opt.manip)));

% sta = state (1 per symbol)
% cho = choice (1/2)
% out = outcome (-1/1)
% cou = counterfactual (-1/1)

% con = condition (≠ state)
% checkmiss = missed response (doesn't matter as LL is computed only if choice = 1 or 2)

% Specificity of S2021c ------------------------------------------------- %
if isfield(db, 'choc'); db.cho = db.choc; end
if isfield(db, 'couc'); db.cou = db.couc; end
if isfield(db, 'outc'); db.out = db.outc; end
if isfield(db, 'stac'); db.sta = db.stac; end
% Specificity of S2021p ------------------------------------------------- %
if isfield(db, 'chop'); db.cho = db.chop; end
if isfield(db, 'outp'); db.out = db.outp; end
if isfield(db, 'stap'); db.sta = db.stap; end
% ----------------------------------------------------------------------- %

init.path_model = fullfile(fileparts(init.path), 'Model'); % Path to save modelisation outputs

%% Start

w = waitbar(0, 'Starting...');

% Initialization
parameters = NaN(numel(db.cho), opt.param_nb, max(opt.whichmodel)); % Nsub * Nparam * Nmodel
ll = NaN(numel(db.cho), max(opt.whichmodel)); % Nsub * Nmodel

initial_point = repmat({NaN(numel(db.cho), opt.param_nb)}, max(opt.whichmodel), 1); % {Nmodel}(Nsub * Nparam)

for sub = 1:numel(db.cho) % Loop through participants

    waitbar(sub/numel(db.cho), w, sprintf('Processing of participant %d / %d', sub, numel(db.cho)));

    % Learning ---------------------------------------------------------- %

    data.learning.cond_idx = db.sta{sub}; % State index (1 to 4)
    data.learning.outcome  = db.out{sub}; % Outcome de l'option choisie

    unchosen_outcomes = NaN(numel(data.learning.outcome), 2);
    choice = db.cho{sub};
    choice(~ismember(choice, [1 2])) = NaN; % If the choice is not 1 or 2 (i.e., missing choice), replace by NaN
    unchosen = 3 - choice; % Pick opposite column = unchosen option

    if isfield(db, 'cou')

        validMask = ~isnan(unchosen); % logical vector: true where unchosen is 1 or 2
        validRows = find(validMask);  % numeric vector of row numbers with valid choices
        validCols = unchosen(validMask);  % same length as validRows
        linIdx = sub2ind(size(unchosen_outcomes), validRows, validCols);

        unchosen_outcomes(linIdx) = db.cou{sub}(validMask);

    end

    data.learning.unchosen_out = unchosen_outcomes; % Toutes les options non choisies
    data.learning.choice_rank = choice; % Which option inside the state (1 or 2)

    %% OPTIMIZATION

    fmin_opt = optimset('Algorithm', 'interior-point', 'Display', 'off', 'MaxIter', 10000, 'MaxFunEval', 10000);
    % The option Display is set to off, which means that the optimization algorithm will run silently, without showing the output of each iteration.
    % The option MaxIter is set to 10000, which means that the algorithm will perform a maximum of 10,000 iterations.

    for model_idx = opt.whichmodel % Loop through models

        model_param = def_models_param(model_idx);

        model_info.idx = model_idx;
        model_info.param_idx = model_param.idx;
        model_info.param_names = model_param.names;

        % For saving
        if sub == numel(db.cho) % Last subject
            fit.in.model_param(model_idx) = model_param;
        end

        % Random point initialization ----------------------------------- %
        % Prepare multiple starting points for estimation

        ddb = model_param.upper_bounds - model_param.lower_bounds; % Where to look for the random point initialization
        ddb(ddb == Inf) = 20; % Replace Inf by 20 (for beta parameter)
        n_param = sum(model_param.idx);

        initial_point_rep = zeros(opt.n_rep, numel(ddb));
        param_rep = NaN(opt.n_rep, numel(ddb));
        ll_rep = NaN(opt.n_rep,1);

        for k_rep = 1:opt.n_rep % Loop through random point initialization

            initial_point_rep(k_rep, model_param.idx) = model_param.lower_bounds(model_param.idx) + rand(1, n_param) .* ddb(model_param.idx);

            % FITTING
            [param_rep(k_rep,:), ll_rep(k_rep)] = fmincon(@(params) ...
                function_model_simulations_mult_options_decay(params, data, model_info, opt.modeling_step, opt.estimation_type, opt.phase_fit),...
                initial_point_rep(k_rep,:), [],[],[],[], model_param.lower_bounds, model_param.upper_bounds, [], fmin_opt);

        end % End of the loop through random point initialization

        % Find the best params over repetitions & store optimization outputs

        [~, best_rep] = min(ll_rep);

        parameters(sub,:,model_idx) = param_rep(best_rep,:);
        ll(sub, model_idx) = ll_rep(best_rep);

        initial_point{model_idx}(sub,:) = initial_point_rep(best_rep,:);

        %% BIC computation

        if strcmp(opt.estimation_type, 'MLE') % BIC should be computed with the maximum‐likelihood estimate (MLE) not the maximum a posteriori estimate (MAP)
            if sub == numel(db.cho) % Last subject

                switch opt.phase_fit
                    case 'learning'
                        total_trial = numel(data.learning.cond_idx);
                end
                fit.out.bic(:,model_idx) = -2 * -ll(:,model_idx) + model_param.nfpm * log(total_trial);

            end
        end

        %% AIC computation

        if strcmp(opt.estimation_type, 'MLE')
            fit.out.aic(:,model_idx) = -2 * -ll(:,model_idx) + model_param.nfpm * 2;
        end

    end % End of the loop through models

    % For backup
    fit.in.data(sub) = data;

end % End of the loop through subjects

delete(w);

if strcmp(opt.estimation_type, 'MLE')
    for model_idx = opt.whichmodel % Loop through models

        fprintf(['\n-------------------------------\n' ...
            'BIC Model %d : %.3f ± %.3f\n' ...
            '-------------------------------\n\n'], model_idx, mean(fit.out.bic(:,model_idx)), std(fit.out.bic(:,model_idx)))
    end
end

%% SAVE DATA

fit.in.opt = opt;
fit.in.fmin_opt = fmin_opt;
fit.in.initial_point = initial_point; % Of the best model for each participants

fit.out.params = parameters;
fit.out.ll = ll;

save(fullfile(init.path_model, opt.estimation_type, sprintf('fit_decay_%s.mat', opt.manip)), 'fit');

%% Model comparison (VBA toolbox)

if numel(opt.whichmodel) > 1 && strcmp(opt.estimation_type, 'MLE')

    vba_opt.verbose = 0;

    % % VBA_groupBMC(-fit.out.ll')
    [posterior,out] = VBA_groupBMC((-fit.out.bic(:,opt.whichmodel)/2)', vba_opt); % Why -1/2 * BIC : https://statswithr.github.io/book/bayesian-model-choice.html
end
