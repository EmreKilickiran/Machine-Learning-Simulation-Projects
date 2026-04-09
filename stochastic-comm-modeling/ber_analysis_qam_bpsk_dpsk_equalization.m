% =========================================================================
% project1_modulation_and_equalization.m
% =========================================================================
% Monte Carlo BER simulation for digital modulation schemes and
% matrix-based channel equalization.
%
% Part A: QAM BER Analysis (4/16/64-QAM vs theoretical bounds)
% Part B: BPSK BER vs data rate (fixed received power)
% Part C: BPSK vs DPSK performance comparison
% Part D: Zero-forcing equalizer design (3-tap and 5-tap)
%
% Author : Yunus Emre Kılıçkıran
% Course : Digital Communication (EE4082), Spring 2024
% =========================================================================

clear; clc; close all;

%% ========== PART A: QAM Modulation (4/16/64-QAM) =======================
fprintf('========== Part A: QAM BER Analysis ==========\n');

M_values  = [4, 16, 64];
EbN0_dB   = 0:2:20;
EbN0      = 10.^(EbN0_dB/10);
numTrials = 10;

simBER_qam  = zeros(length(M_values), length(EbN0_dB));
thBER_qam   = zeros(length(M_values), length(EbN0_dB));

for mIdx = 1:length(M_values)
    M = M_values(mIdx);
    k = log2(M);
    numBits = k * floor(1e6 / k);

    % Theoretical BER (exact)
    thBER_qam(mIdx,:) = (4/k)*(1 - 1/sqrt(M)) .* qfunc(sqrt(3*k*EbN0/(M-1)));

    for i = 1:length(EbN0_dB)
        trialBER = 0;
        for t = 1:numTrials
            bits     = randi([0 1], numBits, 1);
            symbols  = bi2de(reshape(bits,[],k), 'left-msb');
            txSig    = qammod(symbols, M, 'UnitAveragePower', true);

            noisePow = 1 / (2*k*EbN0(i));
            noise    = sqrt(noisePow) * (randn(size(txSig)) + 1j*randn(size(txSig)));
            rxSig    = txSig + noise;

            rxSym    = qamdemod(rxSig, M, 'UnitAveragePower', true);
            rxBits   = reshape(de2bi(rxSym, k, 'left-msb')', [], 1);
            trialBER = trialBER + sum(bits ~= rxBits)/numBits;
        end
        simBER_qam(mIdx,i) = trialBER / numTrials;
    end
end

figure;
colors = {'r','g','b'}; markers = {'o-','s-','d-'};
for mIdx = 1:length(M_values)
    semilogy(EbN0_dB, simBER_qam(mIdx,:), markers{mIdx}, 'LineWidth',1.5, ...
        'DisplayName', sprintf('Simulated %d-QAM', M_values(mIdx))); hold on;
    semilogy(EbN0_dB, thBER_qam(mIdx,:), ['--',colors{mIdx}], 'LineWidth',1.5, ...
        'DisplayName', sprintf('Theoretical %d-QAM', M_values(mIdx)));
end
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('BER: 4-QAM, 16-QAM, 64-QAM'); legend('Location','southwest');
ylim([1e-5 1]);

%% ========== PART B: BPSK BER vs Data Rate ===============================
fprintf('\n========== Part B: BPSK BER vs Data Rate ==========\n');

N0        = 2;
Pr_N0     = 40000;
Rb_values = 3e3:1e3:35e3;
T         = 1e6;

thBER_rate  = zeros(length(Rb_values), 1);
simBER_rate = zeros(length(Rb_values), 1);

for idx = 1:length(Rb_values)
    Rb    = Rb_values(idx);
    Eb    = Pr_N0 / Rb;
    sigma = sqrt(N0/2);

    bits     = randi([0 1], T, 1)*2 - 1;
    received = sqrt(Eb)*bits + sigma*randn(T,1);
    detected = sign(received);
    detected(detected==0) = 1;

    simBER_rate(idx) = sum(detected ~= bits) / T;
    thBER_rate(idx)  = qfunc(sqrt(2*Eb/N0));
end

figure;
semilogy(10*log10(Pr_N0./Rb_values), thBER_rate, '-o', 'LineWidth',1.5, ...
    'DisplayName','Theoretical'); hold on;
semilogy(10*log10(Pr_N0./Rb_values), simBER_rate, '-x', 'LineWidth',1.5, ...
    'DisplayName','Simulated');
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('BPSK BER vs Data Rate'); legend show;

%% ========== PART C: BPSK vs DPSK =======================================
fprintf('\n========== Part C: BPSK vs DPSK ==========\n');

T_c = 100000;
Eb_dB = -4:8;
Eb_vals = 10.^(Eb_dB/10);
N0_c = 2;

simBPSK = zeros(1,length(Eb_vals));  thBPSK = zeros(1,length(Eb_vals));
simDPSK = zeros(1,length(Eb_vals));  thDPSK = zeros(1,length(Eb_vals));
bits_c  = randi([0 1], 1, T_c);

for idx = 1:length(Eb_vals)
    Eb    = Eb_vals(idx);
    sigma = sqrt(N0_c/2);

    % BPSK
    bpsk_sym = 2*bits_c - 1;
    rx_bpsk  = sqrt(Eb)*bpsk_sym + sigma*randn(1,T_c);
    simBPSK(idx) = sum((rx_bpsk > 0) ~= bits_c) / T_c;
    thBPSK(idx)  = qfunc(sqrt(2*Eb/N0_c));

    % DPSK
    dpsk_sym = zeros(1,T_c);  dpsk_sym(1) = 1;
    for i = 2:T_c
        dpsk_sym(i) = dpsk_sym(i-1) * (2*bits_c(i)-1);
    end
    rx_dpsk = sqrt(Eb)*dpsk_sym + sigma*randn(1,T_c);
    dec_dpsk = zeros(1,T_c);
    for i = 2:T_c
        dec_dpsk(i) = rx_dpsk(i)*rx_dpsk(i-1) > 0;
    end
    simDPSK(idx) = sum(dec_dpsk(2:end) ~= bits_c(2:end)) / (T_c-1);
    thDPSK(idx)  = 0.5*exp(-Eb/N0_c);
end

figure;
semilogy(Eb_dB, simBPSK, 'b--', Eb_dB, thBPSK, 'ks', ...
         Eb_dB, simDPSK, 'r-.', Eb_dB, thDPSK, 'mo', 'LineWidth',1.5);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('BPSK vs DPSK'); axis([-4 8 1e-3 1]);
legend('BPSK Sim','BPSK Theory','DPSK Sim','DPSK Theory','Location','southwest');

%% ========== PART D: Zero-Forcing Equalizer ==============================
fprintf('\n========== Part D: Channel Equalization ==========\n');

% --- 3-Tap Equalizer (Channel h) ---
h = [0.1, 0.3, -0.2, 1, 0.4, -0.1, 0.1];

H3 = [1, -0.2, 0.3; 0.4, 1, -0.2; -0.1, 0.4, 1];
d3 = [0; 1; 0];
c3 = H3 \ d3;

fprintf('\n3-Tap Equalizer Coefficients:\n');
fprintf('  c[-1]=%.4f  c[0]=%.4f  c[1]=%.4f\n', c3);

% Equalized output
k_range3 = -3:3;
y_eq3 = zeros(size(k_range3));
for idx = 1:length(k_range3)
    k = k_range3(idx);
    for j = -1:1
        pos = k - j + 4;
        if pos >= 1 && pos <= length(h)
            y_eq3(idx) = y_eq3(idx) + c3(j+2)*h(pos);
        end
    end
end
fprintf('  Equalized output: '); fprintf('%.4f ', y_eq3); fprintf('\n');

% --- 5-Tap Equalizer (Channel h2) ---
h2 = [0.01, 0.02, -0.03, 0.1, 1.0, 0.2, -0.1, 0.05, 0.02];

H5 = [1, -0.03, 0.02, 0.01, 0;
      0.1, 1, -0.03, 0.02, 0.01;
      0.2, 0.1, 1, -0.03, 0.02;
     -0.1, 0.2, 0.1, 1, -0.03;
      0.05,-0.1, 0.2, 0.1, 1];
d5 = [0; 0; 1; 0; 0];
c5 = H5 \ d5;

fprintf('\n5-Tap Equalizer Coefficients:\n');
fprintf('  c[-2]=%.4f  c[-1]=%.4f  c[0]=%.4f  c[1]=%.4f  c[2]=%.4f\n', c5);

k_range5 = -4:4;
y_eq5 = zeros(size(k_range5));
for idx = 1:length(k_range5)
    k = k_range5(idx);
    for j = -2:2
        pos = k - j + 5;
        if pos >= 1 && pos <= length(h2)
            y_eq5(idx) = y_eq5(idx) + c5(j+3)*h2(pos);
        end
    end
end
fprintf('  Equalized output: '); fprintf('%.4f ', y_eq5); fprintf('\n');

fprintf('\nProject 1 complete.\n');
