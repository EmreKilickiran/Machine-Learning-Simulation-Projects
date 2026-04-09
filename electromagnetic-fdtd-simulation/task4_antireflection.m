% =========================================================================
% task4_antireflection.m — Quarter-Wavelength Anti-Reflection Coating
% =========================================================================
% Introduces a matching layer (εr = √(εr1·εr2)) of thickness λ/4
% between two dielectric regions to minimize reflections at 1 GHz.
% Validated via FFT-based frequency-domain analysis of reflected signal.
% =========================================================================

clear; clc; close all;

c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c0^2);

Nz = 600;  dz = 0.5e-3;
z  = (0:Nz-1)*dz;
dt = dz/(2*c0);
Nt = 2000;

%% Material Setup: Region 1 | AR Coating | Region 2
eps_r1 = 1;   eps_r2 = 4;
eps_r_match = sqrt(eps_r1 * eps_r2);   % Geometric mean

f0 = 1e9;
lambda_match = c0 / (f0 * sqrt(eps_r_match));
d_match = lambda_match / 4;            % Quarter-wavelength thickness
match_points = round(d_match / dz);

match_start = 200;
match_end   = match_start + match_points - 1;
right_start = match_end + 1;

fprintf('AR coating εr     : %.2f\n', eps_r_match);
fprintf('AR coating thickness: %.4f m\n', d_match);
fprintf('AR coating points  : %d\n', match_points);

eps_r = eps_r1 * ones(1, Nz);
eps_r(match_start:match_end) = eps_r_match;
eps_r(right_start:end) = eps_r2;
eps   = eps_r * eps0;
sigma = zeros(1, Nz);

%% Source: Raised-Cosine Gaussian Pulse (broadband excitation)
Ez = zeros(1,Nz);  Hy = zeros(1,Nz);
Ez_history = zeros(Nt, Nz);

t0 = 100;  spread = 50;  f_c = 1e9;
source_pos = 20;

for n = 1:Nt
    for k = 1:Nz-1
        Hy(k) = Hy(k) - (dt/(mu0*dz))*(Ez(k+1)-Ez(k));
    end
    for k = 2:Nz-1
        coeff = dt*sigma(k)/(2*eps(k));
        Ez(k) = (1-coeff)/(1+coeff)*Ez(k) ...
              - (dt/(eps(k)*dz))/(1+coeff)*(Hy(k)-Hy(k-1));
    end

    % Raised-cosine Gaussian pulse
    Ez(source_pos) = Ez(source_pos) + ...
        exp(-((n-t0)/spread)^2) * cos(2*pi*f_c*dt*(n-t0));

    Ez(Nz) = Ez(Nz-1);  % ABC

    Ez_history(n,:) = Ez;
end

%% E-field at final time step
figure;
plot(z, Ez);
xlabel('Distance (m)'); ylabel('Ez (V/m)');
title('Ez — Anti-Reflection Matching Layer');

%% FFT of reflected signal at source position
Ez_trace = Ez_history(:, source_pos);

Fs = 1/dt;
L  = length(Ez_trace);
f_axis = Fs*(0:(L/2))/L;

Y  = fft(Ez_trace);
P  = abs(Y/L);
P1 = P(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

figure;
plot(f_axis/1e9, P1, 'b', 'LineWidth', 1.2);
xlim([0 3]);
xlabel('Frequency (GHz)'); ylabel('|Ez(f)|');
title('FFT of Reflected Signal — AR Coating Validation');
xline(1.0, '--r', '1 GHz design freq.', 'LabelVerticalAlignment','top');
