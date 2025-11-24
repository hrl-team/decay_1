%% TWO and FOUR ARMED BANDIT
% -------------------------------------------------------
% Romane Cecchi, 2025
%
% INPUTS:
%   - params: vector or array (optimized by fmincon)
%   - data: structure with fields:
%       - learning.cond_idx: Condition index (1 to 4 ; negative = forced choice)
%       - learning.choice_rank: 1 = worst, 2 = middle, 3 = best choice
%       - learning.outcome: Outcome de l'option choisie
%       - learning.unchosen_out: Valeurs des options non choisies
%       - learning.img_id: Matrix similar to the Q-value matrix (condi * options (from worst to best)) with the corresponding image names
%       - learning.out_fix_ratio: Outcomes fixation time ratio (from worst to best option)
%       - transfer.img_id: Id of symbols presented during the trial
%       - transfer.choice_idx: Position of choice on screen
%       - context.outcomes: Average value of options for each context (condi nb * option nb)
%       - context.variance: Variance of options for each context (condi nb * option nb)
%   - model_info: model number OR structure with fields:
%       - model_info.idx = model number
%       - model_info.param = model parameters (obtained from 'del_models_param.m')
%   - modeling_step: 'fitting'|'simulation'
%   - phase_fit: 'learning'|'transfer'|'both'
%
% OUTPUTS:
%   - lik: Negative log likelihood (from fitting) -> Used by fmincon
%   - choice_proba: Structure with fields:
%       - learning: Probability of choosing each symbol (from worst to best)
%       - transfer_right_opt: Probability of choosing the option on the right in the transfer test
%   - Q: Q-value matrix (condi * options (worst to best))
%
% MODELS: (no normalization)
% 1: 1 alpha (α± = α)
% 2: 2 alpha (α_c+ = α_u- and α_c- = α_u+)
% 3: 4 alpha (chosen/unchosen x +/-)
% 4: Bayes-greedy (α = deterministic = 1/(t+3))
% 5: Simple decay (1 alpha + 1 delta)
% 6: Full decay (2 alpha [as in model 2] + 2 delta [+/-])

function [lik, out] = function_model_simulations_mult_options_decay(params, data, model_info, modeling_step, estimation_type, phase_fit)

% Parameters (optimized by fmincon when fitting) /!\ Must corresponds to "def_models_param.m" names
beta      = params(1); % choice temperature
alpha_p_c = params(2); % alpha + chosen option
alpha_m_c = params(3); % alpha - chosen options
alpha_p_u = params(4); % alpha + unchosen option
alpha_m_u = params(5); % alpha - unchosen options
decay     = params(6);
decay_p   = params(7);
decay_m   = params(8);

% Data
s = data.learning.cond_idx; % State = Condition index (1 to 4 ; negative = forced choice)

switch modeling_step
    case {'fitting', 'prediction'} % Model fitting

        a  = data.learning.choice_rank; % Choice option (1 ou 2)
        r  = data.learning.outcome; % Outcome de l'option choisie (i.e., reward received)

        % Valeurs des options non choisies du LEARNING (de la plus petite (idx 1) à la plus grande (idx 2 ou 4))
        numTrials = size(data.learning.unchosen_out, 1); % Number of trials in learning
        maxUnchosen = max(sum(~isnan(data.learning.unchosen_out), 2));  % Determine the maximum number of unchosen options across all trials

        if maxUnchosen == 0 % Partial feedback

            CF = data.learning.unchosen_out;

        else % Complete feedback (or at least some feedback from unchosen options)

            % CF = NaN(numTrials, maxUnchosen);

            % for t = 1:numTrials
            %     unchosen_out = sort(data.learning.unchosen_out(t, ~isnan(data.learning.unchosen_out(t, :)))); % Extract and sort the unchosen outcomes for the current trial
            %     CF(t,1:numel(unchosen_out)) = unchosen_out;
            % end

            M = data.learning.unchosen_out; % numTrials x numberOfOptions
            M(isnan(M)) = inf; % Replace NaNs with +Inf so they sort to the end of each row
            M_sorted = sort(M, 2, 'ascend'); % Sort within each row (dimension 2), rows (trials) stay in the same order
            M_sorted(isinf(M_sorted)) = NaN; % Put back NaNs where +Inf are

            % Keep only the first maxUnchosen columns
            CF = M_sorted(:, 1:maxUnchosen);

        end

    case {'simulation'} % Model simulation

        a = NaN(numel(s),1); % State action = Simulated choice
        r = NaN(numel(s),1); % Outcome de l'option choisie (i.e., reward received)
        CF = NaN(numel(s), size(data.context.outcomes,2)-1); % Valeurs des options non choisies du LEARNING (de la plus petite (idx 1) à la plus grande (idx 2 ou 4))

end

nb_choice_opt = NaN(numel(unique(s)),1);
for cont = unique(s)'
    nb_choice_opt(cont) = max(a(s == cont));
end

nb_condi = max(data.learning.cond_idx);

% Model_info can be a structure or a scalar
if isstruct(model_info)
    model_idx = model_info.idx;
else
    model_idx = model_info;
end

% Make 'phase_fit' an optional parameter (as we don't need it for the simulation step)
if ~exist('phase_fit','var')
    phase_fit = '';
end

switch modeling_step
    case {'simulation', 'prediction'}
        choice_proba.learning = NaN(length(data.learning.cond_idx), max(nb_choice_opt));
end

% Model initialization
% Expected value (initialized to 50/25 if model = 1 and 0.5 otherwise) -> condi * options (worst to best)

Q = NaN(nb_condi, max(nb_choice_opt)); % Initialize the matrix with NaNs

% Assign zeros based on nb_choice_opt
for i = 1:nb_condi
    Q(i, 1:nb_choice_opt(i)) = 0; % Initialization at 0
end

lik = 0; % Log-likelihood = log of a softmax function (==> Output of fmincon)

t_state = zeros(nb_condi,1); % For model 4 (Bayes-greedy)
alpha_state = repmat(alpha_p_c, nb_condi, 1); % For model 5 (simple decay)
alpha_p_c_state = repmat(alpha_p_c, nb_condi, 1); % For model 6 (full decay)
alpha_m_c_state = repmat(alpha_m_c, nb_condi, 1); % For model 6 (full decay)

for t = 1:numel(s) % Loop through learning trials

    t_state(s(t)) = t_state(s(t)) + 1;

    if ~isnan(a(t))

        Q_all = Q(s(t),:); % Q-values of all options of the condition
        Q_all(isnan(Q_all)) = []; % Remove NaNs

        % --------------------------------------------------------------- %

        switch modeling_step
            case {'simulation', 'prediction'} % Model simulation (proba de choisir telle ou telle option)

                % Softmax decision rule

                if s(t) > 0 % Free choice (100% ternary)

                    % Probability of choosing each symbol
                    % proba = exp(beta * Q_all) / sum(exp(beta * Q_all)); % From worst to best option

                    % Numerically stable softmax
                    Q_scaled = beta * Q_all;              % [50, 100, 150] → this can overflow
                    Q_stable = Q_scaled - max(Q_scaled);  % [-100, -50, 0]
                    proba = exp(Q_stable) / sum(exp(Q_stable));

                end

                if strcmp(modeling_step, 'simulation')

                    % Make the choice based on the probability of choosing each symbol
                    choice_opt = 1:numel(Q_all); % Choice options (e.g.: 1 = min, 2 = mid, 3 = max)

                    a(t) = choice_opt(find(rand < cumsum(proba), 1, 'first')); % Select a symbol with its probability

                end

                choice_proba.learning(t,1:numel(proba)) = proba; % Probability of choosing each symbol (from worst to best)

        end % End of modeling step condition

        % Get Q-values -------------------------------------------------- %

        out.Q_time(:,:,t) = Q;

        Q_chosen = Q(abs(s(t)), a(t)); % State-action value

        choice_opt_idx = 1:nb_choice_opt(abs(s(t)));
        unchosen_idx = sort(choice_opt_idx(choice_opt_idx ~= a(t))); % Indexes of unchosen options (sorted)

        Q_unchosen = Q(abs(s(t)), unchosen_idx); % From smallest to highest value

        % --------------------------------------------------------------- %

        if strcmp(modeling_step, 'fitting') % Model fitting (compute LL)
            if any(strcmp(phase_fit, {'learning', 'both'})) && ~isnan(a(t)) % Fit including learning phase

                % Log-likelihood function: sum over trials of the log of the choice probability (i.e., softmax function)
                lik = lik + (beta * Q_chosen - log(sum(exp(beta * Q_all))));

            end
        end % End of modeling step condition

        %% Learning steps

        s(t) = abs(s(t));

        switch modeling_step
            case {'simulation'} % Model simulation

                % For each trial, set the reward behind each symbol
                mean_out = data.context.outcomes(s(t),:);
                var_out = data.context.variance(s(t),:);
                outcomes = round(normrnd(mean_out, var_out));

                % Constrain outcomes from 0 to 100
                outcomes(outcomes > 100) = 100;
                outcomes(outcomes < 0) = 0;

                % Define outcomes
                r(t) = outcomes(a(t));
                CF(t,1:numel(unchosen_idx)) = outcomes(unchosen_idx);

        end

        % Update values with reward
        % all_opt_val = [r(t), CF(t,:)]; % Outcomes of all options
        % all_opt_val(isnan(all_opt_val)) = []; % Remove NaNs

        % range_norm_fun = @(val, range) (val - min(range)) / (max(range) - min(range));

        if model_idx == 1 % 1 alpha (α± = α)

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;
            Q(s(t), a(t)) = Q_chosen + alpha_p_c * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);
                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_p_c * deltaC; % Update Q-value of unchosen options

                end
            end

        elseif model_idx == 2 % 2 alpha (α_c+ = α_u- and α_c- = α_u+)

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;

            if deltaF >= 0 % Positive prediction error
                alpha_spec = alpha_p_c;
            elseif deltaF < 0 % Negative prediction error
                alpha_spec = alpha_m_c;
            end

            Q(s(t), a(t)) = Q_chosen + alpha_spec * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);

                    if deltaC >= 0 % Positive prediction error
                        alpha_spec = alpha_m_c;
                    elseif deltaC < 0 % Negative prediction error
                        alpha_spec = alpha_p_c;
                    end

                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_spec * deltaC; % Update Q-value of unchosen options

                end
            end

        elseif model_idx == 3 % 4 alpha (chosen/unchosen x +/-)

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;

            if deltaF >= 0 % Positive prediction error
                alpha_spec = alpha_p_c;
            elseif deltaF < 0 % Negative prediction error
                alpha_spec = alpha_m_c;
            end

            Q(s(t), a(t)) = Q_chosen + alpha_spec * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);

                    if deltaC >= 0 % Positive prediction error
                        alpha_spec = alpha_p_u;
                    elseif deltaC < 0 % Negative prediction error
                        alpha_spec = alpha_m_u;
                    end

                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_spec * deltaC; % Update Q-value of unchosen options

                end
            end

        elseif model_idx == 4 % Bayes-greedy (α = deterministic = 1/(t+3))

            alpha_bayes = 1/(t_state(s(t)) + 3);

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;
            Q(s(t), a(t)) = Q_chosen + alpha_bayes * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);
                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_bayes * deltaC; % Update Q-value of unchosen options

                end
            end

        elseif model_idx == 5 % Simple decay (1 alpha + 1 delta)

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;
            Q(s(t), a(t)) = Q_chosen + alpha_state(s(t)) * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);
                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_state(s(t)) * deltaC; % Update Q-value of unchosen options

                end
            end

            % Decay alpha
            alpha_state(s(t)) = alpha_state(s(t)) * decay;

        elseif model_idx == 6 % Full decay (2 alpha [α_c+ = α_u- and α_c- = α_u+] + 2 delta [+/-])

            % Chosen option (factual)
            deltaF = r(t) - Q_chosen;

            if deltaF >= 0 % Positive prediction error
                alpha_spec = alpha_p_c_state(s(t));
            elseif deltaF < 0 % Negative prediction error
                alpha_spec = alpha_m_c_state(s(t));
            end

            Q(s(t), a(t)) = Q_chosen + alpha_spec * deltaF; % Update Q-value of chosen option

            % Unchosen options (counterfactual)

            if maxUnchosen ~= 0 % If there is a counterfactual
                for c = 1:numel(Q_unchosen) % Loop through unchosen options (from lowest to highest)

                    deltaC  = CF(t,c) - Q_unchosen(c);

                    if deltaC >= 0 % Positive prediction error
                        alpha_spec = alpha_m_c_state(s(t));
                    elseif deltaC < 0 % Negative prediction error
                        alpha_spec = alpha_p_c_state(s(t));
                    end

                    Q(s(t), unchosen_idx(c)) = Q_unchosen(c) + alpha_spec * deltaC; % Update Q-value of unchosen options

                end
            end

            % Decay alpha
            alpha_p_c_state(s(t)) = alpha_p_c_state(s(t)) * decay_p;
            alpha_m_c_state(s(t)) = alpha_m_c_state(s(t)) * decay_m;

        end % End of model definition

    end % End of the condition if ~isnan(a(t))
end % End of the loop through learning trials

%% End of task steps

lik = -lik; % Must be inverted as fmincon searches for the minimum value

%% Prior penalization (Maximum A Posteriori (MAP))
% See Daw et al., 2011 - Model-based influences on humans' choices and striatal prediction errors
% Gershman, 2016 - Empirical priors for reinforcement learning models

if strcmp(estimation_type, 'MAP')

    switch modeling_step
        case {'fitting'} % Model fitting

            count = 0;
            p = NaN(1,sum(model_info.param_idx));

            for i = find(model_info.param_idx)
                count = count + 1;
                switch model_info.param_names{i}
                    case 'beta'
                        p(count) = log(gampdf(params(i),1.2,5));
                    case {'alpha_plus_chosen', 'alpha_minus_chosen', 'alpha_plus_unchosen', 'alpha_minus_unchosen', 'decay', 'decay_plus', 'decay_minus'}
                        p(count) = log(betapdf(params(i),1.1,1.1)); % The beta probability density function is bounded (compared to the gamma one)
                end
            end

            p = -sum(p);

            lik = p + lik;

    end
end

%% Output

switch modeling_step
    case {'simulation', 'prediction'}
        out.choice_proba = choice_proba;
end

out.Q_final = Q;

if strcmp(modeling_step, 'simulation')
    out.data.learning.choice_rank = a;
    out.data.learning.outcome = r;
    out.data.learning.unchosen_out = CF;
    out.data.transfer.choice_idx = aa;
end

end

%% Helper functions

