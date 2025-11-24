
clear
close all
clc

% MODELS: (no normalization)
% 1: 1 alpha (α± = α)
% 2: 2 alpha (α_c+ = α_u- and α_c- = α_u+)
% 3: 4 alpha (chosen/unchosen x +/-) -> Only for complete tasks
% 4: Bayes-greedy (α = deterministic = 1/(t+3))
% 5: Simple decay (1 alpha + 1 delta)
% 6: Full decay (2 alpha [as in model 2] + 2 delta [+/-])

% /!\ Based on 'model_fitting_decay.m' results
% CHECK THAT THE CORRECT VERSION HAS BEEN RUN

opt.path = '/Users/romane/Documents/BDD_Decay_alpha/Model';
opt.manip = {'L2017a', 'L2017b', 'P2017a', 'P2017b', 'S2021c', 'S2021p'};

opt.model_nb.conf = 2;
opt.model_nb.bayes = 4;
opt.model_nb.simple_decay = 5;
opt.model_nb.full_decay = 6;

%% Figure 1: Δ{conf - bayes greedy} x Δ{conf - simple decay}

fig = figure;
tiledlayout('horizontal')

for manip = 1:numel(opt.manip) % Loop through experiments

    nexttile
    hold on

    clearvars -except opt manip fig
    load(fullfile(opt.path, 'MLE', sprintf('fit_decay_%s.mat', opt.manip{manip})))

    % Individual points
    delta_bayes_conf = fit.out.bic(:,opt.model_nb.bayes) - fit.out.bic(:,opt.model_nb.conf);
    delta_simple_decay_conf = fit.out.bic(:,opt.model_nb.simple_decay) - fit.out.bic(:,opt.model_nb.conf);

    x = delta_bayes_conf;
    y = delta_simple_decay_conf;

    scatter(x, y, 'filled')

    % Mean
    mean_x = mean(x);
    sem_x = std(x)/sqrt(numel(x));

    mean_y = mean(y);
    sem_y = std(y)/sqrt(numel(y));

    scatter(mean_x, mean_y, 'r', 'filled')
    errorbar(mean_x, mean_y, sem_x, 'r')
    errorbar(mean_x, mean_y, sem_y, 'r', 'horizontal')

    % Limits: make them symmetric around 0
    Xlim = max(abs(xlim));
    Ylim = max(abs(ylim));

    xlim([-Xlim Xlim]);
    ylim([-Ylim Ylim]);

    % Participant number in each quadrant
    Q = [sum(x<0 & y>=0), sum(x>=0 & y>=0);
        sum(x<0 & y<0), sum(x>=0 & y<0)];

    if sum(Q(:)) ~= numel(x)
        keyboard
    end

    % Choose positions roughly at the center of each quadrant
    xNeg = -Xlim * 0.8;   % center of left half
    xPos =  Xlim * 0.2;   % center of right half
    yPos =  Ylim * 0.9;   % center of top half
    yNeg = -Ylim * 0.9;   % center of bottom half

    text(xNeg, yPos, sprintf('n = %d', Q(1,1)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold'); % Top-left (Q(1,1))
    text(xPos, yPos, sprintf('n = %d', Q(1,2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold'); % Top-right (Q(1,2))
    text(xNeg, yNeg, sprintf('n = %d', Q(2,1)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold'); % Bottom-left (Q(2,1))
    text(xPos, yNeg, sprintf('n = %d', Q(2,2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold'); % Bottom-right (Q(2,2))

    % Lines at zero
    plot(xlim, zeros(1,2), 'k--')
    plot(zeros(1,2), ylim, 'k--')

    % Labels
    xlabel('Δ{bayes greedy - conf}');
    ylabel('Δ{simple decay - conf}');
    title(sprintf('Experiment: %s', opt.manip{manip}));

end % End of the loop through experiments

close(fig)

%% Figure 2: Parameters of the full decay model

fig = figure;
t = tiledlayout(1,12);

sem = @(x) std(x)/sqrt(numel(x));

for manip = 1:numel(opt.manip) % Loop through experiments

    clearvars -except opt manip fig sem out
    load(fullfile(opt.path, 'MAP', sprintf('fit_decay_%s.mat', opt.manip{manip})))

    % Get parameters
    params = fit.out.params(:,:,opt.model_nb.full_decay);
    alpha_plus = params(:,strcmp(fit.in.model_param(opt.model_nb.full_decay).names, {'alpha_plus_chosen'}));
    alpha_minus = params(:,strcmp(fit.in.model_param(opt.model_nb.full_decay).names, {'alpha_minus_chosen'}));
    decay_plus = params(:,strcmp(fit.in.model_param(opt.model_nb.full_decay).names, {'decay_plus'}));
    decay_minus = params(:,strcmp(fit.in.model_param(opt.model_nb.full_decay).names, {'decay_minus'}));

    % Plot alpha+ and alpha- -------------------------------------------- %
    nexttile(1, [1 2])
    hold on

    errorbar([1 2], [mean(alpha_plus) mean(alpha_minus)], [sem(alpha_plus), sem(alpha_minus)], '-ko');
    text(1 - 0.1, mean(alpha_plus), opt.manip{manip}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    out.lr(manip,:) = [mean(alpha_plus) mean(alpha_minus)];

    xticks([1 2])
    xticklabels({'\alpha+', '\alpha-'})
    xlim([.6 2.4])
    ylim([0 1])

    title('Learning rates')

    % Difference in alpha+ and alpha- ----------------------------------- %
    nexttile(3)
    hold on

    errorbar(1, mean(alpha_plus-alpha_minus), sem(alpha_plus-alpha_minus), 'ko');
    out.delta_lr(manip) = mean(alpha_plus-alpha_minus);

    xticks(1)
    xticklabels({'\alpha+ - \alpha-'})
    xlim([.6 1.4])
    ylim([-.5 .5])

    title('\Delta Learning rates')

    % Plot decay rates -------------------------------------------------- %
    nexttile(4, [1 2])
    hold on

    errorbar([1 2], [mean(decay_plus) mean(decay_minus)], [sem(decay_plus), sem(decay_minus)], '-ko');
    text(1 - 0.1, mean(decay_plus), opt.manip{manip}, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', 12);
    out.decay(manip,:) = [mean(decay_plus) mean(decay_minus)];

    xticks([1 2])
    xticklabels({'\delta+', '\delta-'})
    xlim([.6 2.4])
    ylim([0 1])

    title('Decays')

    % Difference in decay+ and decay- ----------------------------------- %
    nexttile(6)
    hold on

    errorbar(1, mean(decay_plus-decay_minus), sem(decay_plus-decay_minus), 'ko');
    out.delta_decay(manip) = mean(decay_plus-decay_minus);

    xticks(1)
    xticklabels({'\delta+ - \delta-'})
    xlim([.6 1.4])
    ylim([-.5 .5])

    title('\Delta decays')

    % Alpha+ temporal evolution ----------------------------------------- %
    nexttile(7, [1 3])
    hold on

    v0 = alpha_plus; % starting value
    c  = decay_plus; % multiplier
    n  = unique(arrayfun(@(s) unique(groupcounts(s.learning.cond_idx)), fit.in.data)); % length of vector = number of trials per states

    alpha_plus_time = v0 .* c.^(0:n-1); % subj x trial

    % Plot the mean across participants
    x = mean(alpha_plus_time);
    plot(x, 'k', 'LineWidth', .5)
    text(n + 0.2, x(end), opt.manip{manip}, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12);

    out.alpha_plus_time{manip,:} = x; % manip x trial

    % Limits
    out.xlim(manip,:) = n;
    out.ylim(manip,:) = round(x(1),1);

    xlim([1 max(out.xlim)])
    ylim([0 max(out.ylim)])

    title('Positive learning rate evolution (\alpha+ \times \delta+)')

    % Alpha- temporal evolution ----------------------------------------- %
    nexttile(10, [1 3])
    hold on

    v0 = alpha_minus; % starting value
    c  = decay_minus; % multiplier
    n  = unique(arrayfun(@(s) unique(groupcounts(s.learning.cond_idx)), fit.in.data)); % length of vector = number of trials per states

    alpha_minus_time = v0 .* c.^(0:n-1); % subj x trial

    % Plot the mean across participants
    x = mean(alpha_minus_time);
    plot(x, 'k', 'LineWidth', .5)
    text(n + 0.2, x(end), opt.manip{manip}, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 12);

    out.alpha_minus_time{manip,:} = x; % manip x trial

    % Limits
    xlim([1 max(out.xlim)])
    ylim([0 max(out.ylim)])

    title('Negative learning rate evolution (\alpha- \times \delta-)')

end % End of the loop through experiments

% Add means through experiments ----------------------------------------- %

nexttile(1, [1 2])
errorbar([1 2], mean(out.lr), std(out.lr)/sqrt(size(out.lr, 1)), '-ro');

nexttile(4, [1 2])
errorbar([1 2], mean(out.decay), std(out.decay)/sqrt(size(out.decay, 1)), '-ro');

nexttile(3)
errorbar(1, mean(out.delta_lr), sem(out.delta_lr), 'ro');

nexttile(6)
errorbar(1, mean(out.delta_decay), sem(out.delta_decay), 'ro');

nexttile(7, [1 3])
lens = cellfun(@numel, out.alpha_plus_time);
Lmax = max(lens);
% padded = cell2mat(cellfun(@(v) [v nan(1, Lmax-numel(v))], out.alpha_plus_time, 'UniformOutput', false)); % Pad with NaNs
padded = cell2mat(cellfun(@(v) [v repmat(v(end), 1, Lmax-numel(v))], out.alpha_plus_time, 'UniformOutput', false)); % Pad with the final value
plot(mean(padded), 'r', 'LineWidth', 2);

nexttile(10, [1 3])
lens = cellfun(@numel, out.alpha_minus_time);
Lmax = max(lens);
% padded = cell2mat(cellfun(@(v) [v nan(1, Lmax-numel(v))], out.alpha_minus_time, 'UniformOutput', false)); % Pad with NaNs
padded = cell2mat(cellfun(@(v) [v repmat(v(end), 1, Lmax-numel(v))], out.alpha_minus_time, 'UniformOutput', false)); % Pad with the final value
plot(mean(padded), 'r', 'LineWidth', 2);

close(fig)
