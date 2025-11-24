% OUTPUTS :
% - param.lower_bounds
% - param.upper_bounds
% - param.initial_point
% - param.nfpm (number of free parameters)

% MODELS: (no normalization)
% 1: 1 alpha (α± = α)
% 2: 2 alpha (α_c+ = α_u- and α_c- = α_u+)
% 3: 4 alpha (chosen/unchosen x +/-)
% 4: Bayes-greedy (α = deterministic = 1/(t+3))
% 5: Simple decay (1 alpha + 1 delta)
% 6: Full decay (2 alpha [as in model 2] + 2 delta [+/-])

function param = def_models_param(model_idx)

% Parameters:
% 1 = beta (choice temperature)
% 2 = alpha + chosen option
% 3 = alpha - chosen options
% 4 = alpha + unchosen options
% 5 = alpha - unchosen options

param.names = {'beta', 'alpha_plus_chosen', 'alpha_minus_chosen', 'alpha_plus_unchosen', 'alpha_minus_unchosen', 'decay', 'decay_plus', 'decay_minus'};

param.lower_bounds = zeros(1, numel(param.names));

% Initialization
param.initial_point = NaN(1, numel(param.names));
param.upper_bounds = NaN(1, numel(param.names));

% Common to all models
param.initial_point(ismember(param.names, {'beta'})) = 1;
param.upper_bounds(ismember(param.names, {'beta'})) = Inf;

% Model specifics
switch model_idx
    case {1}
        param.initial_point(ismember(param.names, {'alpha_plus_chosen'})) = .5;
        param.upper_bounds(ismember(param.names, {'alpha_plus_chosen'})) = 1;
    case {2}
        param.initial_point(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen'})) = .5;
        param.upper_bounds(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen'})) = 1;
    case {3}
        param.initial_point(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen', 'alpha_plus_unchosen', 'alpha_minus_unchosen'})) = .5;
        param.upper_bounds(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen', 'alpha_plus_unchosen', 'alpha_minus_unchosen'})) = 1;
    case {5}
        param.initial_point(ismember(param.names, {'alpha_plus_chosen', 'decay'})) = .5;
        param.upper_bounds(ismember(param.names, {'alpha_plus_chosen', 'decay'})) = 1;
    case {6}
        param.initial_point(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen', 'decay_plus', 'decay_minus'})) = .5;
        param.upper_bounds(ismember(param.names, {'alpha_plus_chosen', 'alpha_minus_chosen', 'decay_plus', 'decay_minus'})) = 1;
end

%% Position of "active" parameters

param.idx = ~isnan(param.initial_point);

%% Number of free parameters

param.nfpm = sum(~isnan(param.initial_point));

%% Replace NaN with 0 (for fmincon)

param.initial_point(isnan(param.initial_point)) = 0;
param.upper_bounds(isnan(param.upper_bounds)) = 0;

end