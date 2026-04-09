% =========================================================================
% project2_convolutional_coding.m
% =========================================================================
% Monte Carlo BER simulation for convolutional codes with Viterbi decoding.
%
% Part A: Uncoded BPSK vs coded (Rate 1/2, 1/3) — coding gain analysis
% Part B: Constraint length sweep (K=3,5,7,9) — accuracy vs complexity
% Part C: BER upper bounds vs simulated (distance spectrum analysis)
% Part D: Execution time benchmarking (5s → 21s for 5M samples)
% Part E: Hard decision (HDD) vs soft decision decoding (SDD)
%
% Author : Yunus Emre Kılıçkıran
% Course : Digital Communication (EE4082), Spring 2024
% =========================================================================

clear; clc; close all;

%% ========== PART A: Coding Gain (Rate 1/2 vs 1/3) ======================
fprintf('========== Part A: Uncoded vs Coded BPSK ==========\n');

N_bits  = 1e7;
EbN0_dB = 0:10;
EbN0    = 10.^(EbN0_dB/10);

K = 3;
trellis_12 = poly2trellis(K, [7 5]);       % Rate 1/2
trellis_13 = poly2trellis(K, [7 7 5]);     % Rate 1/3

data = randi([0 1], 1, N_bits);

BER_uncoded = zeros(1, length(EbN0_dB));
BER_12      = zeros(1, length(EbN0_dB));
BER_13      = zeros(1, length(EbN0_dB));

coded_12 = convenc(data, trellis_12);
coded_13 = convenc(data, trellis_13);

for idx = 1:length(EbN0)
    snr = EbN0(idx);
    noise_std = sqrt(1/(2*snr));

    % Uncoded BPSK
    tx = 2*data - 1;
    rx = tx + noise_std*randn(1, N_bits);
    BER_uncoded(idx) = sum((rx>0) ~= data) / N_bits;

    % Rate 1/2
    tx12 = 2*coded_12 - 1;
    rx12 = tx12 + noise_std*randn(1, length(tx12));
    dec12 = vitdec(rx12>0, trellis_12, 50, 'trunc', 'hard');
    BER_12(idx) = sum(dec12 ~= data) / N_bits;

    % Rate 1/3
    tx13 = 2*coded_13 - 1;
    rx13 = tx13 + noise_std*randn(1, length(tx13));
    dec13 = vitdec(rx13>0, trellis_13, 50, 'trunc', 'hard');
    BER_13(idx) = sum(dec13 ~= data) / N_bits;
end

figure;
semilogy(EbN0_dB, BER_uncoded, 'k-o', EbN0_dB, BER_12, 'r-s', ...
         EbN0_dB, BER_13, 'b-d', 'LineWidth', 1.5);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('Uncoded vs Coded BPSK'); axis([0 10 1e-6 1]);
legend('Uncoded','Rate 1/2','Rate 1/3','Location','southwest');

%% ========== PART B: Constraint Length Sweep (K=3,5,7,9) =================
fprintf('\n========== Part B: Constraint Length Sweep ==========\n');

K_values   = [3, 5, 7, 9];
genPolys   = {[7 5], [23 25], [117 155], [657 435]};
numBits_B  = 1e7;
BER_K      = zeros(length(K_values), length(EbN0_dB));

for kIdx = 1:length(K_values)
    trellis = poly2trellis(K_values(kIdx), genPolys{kIdx});
    data_B  = randi([0 1], numBits_B, 1);
    coded   = convenc(data_B, trellis);

    for idx = 1:length(EbN0_dB)
        snr = 10^(EbN0_dB(idx)/10);
        tx  = 2*coded - 1;
        rx  = tx + sqrt(1/(2*snr))*randn(size(tx));
        dec = vitdec(rx>0, trellis, 5*K_values(kIdx), 'trunc', 'hard');
        BER_K(kIdx, idx) = sum(dec ~= data_B) / numBits_B;
    end
end

figure;
for kIdx = 1:length(K_values)
    semilogy(EbN0_dB, BER_K(kIdx,:), 'o-', 'LineWidth',2, ...
        'DisplayName', sprintf('K = %d', K_values(kIdx))); hold on;
end
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('BER vs K (Rate 1/2 Convolutional Codes)'); legend show;

%% ========== PART C: BER Upper Bounds ====================================
fprintf('\n========== Part C: BER Upper Bounds ==========\n');

K_ub = [3, 5, 7];
trellisUB = {
    poly2trellis(3, {'1+x+x^2', '1+x^2'}),
    poly2trellis(5, {'1+x^2+x^3+x^4', '1+x+x^4'}),
    poly2trellis(7, {'1+x^3+x^4+x^5+x^6', '1+x+x^3+x^4+x^6'})
};
EbN0_ub   = 2:8;
numBits_C = 5e6;
BER_sim_ub = zeros(length(K_ub), length(EbN0_ub));
BER_bound  = zeros(length(K_ub), length(EbN0_ub));

for kIdx = 1:length(K_ub)
    trellis  = trellisUB{kIdx};
    distSpec = distspec(trellis, 10);
    data_C   = randi([0 1], numBits_C, 1);
    coded    = convenc(data_C, trellis);

    for idx = 1:length(EbN0_ub)
        snr = 10^(EbN0_ub(idx)/10);
        tx  = 2*coded - 1;
        rx  = tx + sqrt(1/(2*snr))*randn(size(tx));
        dec = vitdec(rx>0, trellis, 10*K_ub(kIdx), 'trunc', 'hard');
        BER_sim_ub(kIdx,idx) = sum(dec ~= data_C) / numBits_C;
        BER_bound(kIdx,idx)  = bercoding(EbN0_ub(idx), 'conv', 'hard', 0.5, distSpec);
    end
end

figure;
for kIdx = 1:length(K_ub)
    semilogy(EbN0_ub, BER_sim_ub(kIdx,:), 'o-', 'LineWidth',2, ...
        'DisplayName', sprintf('Simulated K=%d', K_ub(kIdx))); hold on;
    semilogy(EbN0_ub, BER_bound(kIdx,:), '--', 'LineWidth',2, ...
        'DisplayName', sprintf('Upper Bound K=%d', K_ub(kIdx)));
end
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('Simulated BER vs Analytical Upper Bounds'); legend show;

%% ========== PART D: Execution Time Benchmarking =========================
fprintf('\n========== Part D: Viterbi Execution Time ==========\n');

execTimes = zeros(length(K_ub), 1);
for kIdx = 1:length(K_ub)
    trellis = trellisUB{kIdx};
    data_D  = randi([0 1], numBits_C, 1);
    coded   = convenc(data_D, trellis);
    tx = 2*coded - 1;
    rx = tx + 0.1*randn(size(tx));

    tic;
    vitdec(rx>0, trellis, 5*K_ub(kIdx), 'trunc', 'hard');
    execTimes(kIdx) = toc;
    fprintf('  K=%d: %.2f s\n', K_ub(kIdx), execTimes(kIdx));
end

figure;
bar(K_ub, execTimes);
grid on; xlabel('Constraint Length (K)'); ylabel('Time (s)');
title('Viterbi Decoder Execution Time (5M samples)');

%% ========== PART E: HDD vs SDD (K=3) ===================================
fprintf('\n========== Part E: Hard vs Soft Decision Decoding ==========\n');

trellis_e = poly2trellis(3, {'1+x+x^2', '1+x^2'});
distSpec_e = distspec(trellis_e, 10);
thresh = [-0.5 0 0.5];

berHDD = zeros(1,length(EbN0_ub));  berSDD = zeros(1,length(EbN0_ub));
ubHDD  = zeros(1,length(EbN0_ub));  ubSDD  = zeros(1,length(EbN0_ub));
numBits_E = 5e6;
data_E = randi([0 1], numBits_E, 1);
coded_E = convenc(data_E, trellis_e);

for idx = 1:length(EbN0_ub)
    snr = 10^(EbN0_ub(idx)/10);
    tx  = 2*coded_E - 1;
    rx  = tx + sqrt(1/(2*snr))*randn(size(tx));
    tbdepth = 30;

    % Hard decision
    decHDD = vitdec(rx>0, trellis_e, tbdepth, 'trunc', 'hard');
    berHDD(idx) = sum(decHDD ~= data_E) / numBits_E;

    % Soft decision (4-level quantization)
    qData   = quantiz(rx, thresh);
    softIn  = reshape(qData, size(coded_E));
    decSDD  = vitdec(softIn, trellis_e, tbdepth, 'trunc', 'soft', 2);
    decSDD  = decSDD(1:numBits_E);
    berSDD(idx) = sum(decSDD ~= data_E) / numBits_E;

    ubHDD(idx) = bercoding(EbN0_ub(idx), 'conv', 'hard', 0.5, distSpec_e);
    ubSDD(idx) = bercoding(EbN0_ub(idx), 'conv', 'soft', 0.5, distSpec_e);
end

figure;
semilogy(EbN0_ub, berHDD, 'o-', EbN0_ub, berSDD, 'x-', ...
         EbN0_ub, ubHDD, '--', EbN0_ub, ubSDD, '--', 'LineWidth', 2);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('HDD vs SDD (K=3, Rate 1/2)');
legend('HDD Sim','SDD Sim','HDD Bound','SDD Bound','Location','southwest');

fprintf('\nProject 2 complete.\n');
