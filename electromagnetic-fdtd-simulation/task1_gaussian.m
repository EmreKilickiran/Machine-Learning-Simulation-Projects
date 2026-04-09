% =========================================================================
% task1_gaussian.m — Free Space Wave Propagation (Gaussian Pulse Source)
% =========================================================================
% Simulates a raised-cosine Gaussian pulse propagating in free space.
% Validates velocity via cross-correlation-based peak tracking.
% =========================================================================

clear; clc; close all;

%% Physical Constants
c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0 * c0^2);

%% Grid & Time Parameters
Nz = 3000;
dz = 0.5e-3;
z  = (0:Nz-1)*dz;
dt = dz/(2*c0);
Nt = 20000;

%% Field Initialization
Ez = zeros(1, Nz);
Hy = zeros(1, Nz);
Ez_history = zeros(Nt, Nz);
Hy_history = zeros(Nt, Nz);

%% Source: Raised-Cosine Gaussian Pulse
f     = 1e9;
omega = 2*pi*f;
source_pos = 2;
t = (0:Nt-1)*dt;

t0  = 6e-9;    % Pulse center
tau = 3e-9;    % Pulse width

%% FDTD Loop
for n = 1:Nt
    for k = 1:Nz-1
        Hy(k) = Hy(k) - (dt/(mu0*dz)) * (Ez(k+1) - Ez(k));
    end

    for k = 2:Nz-1
        Ez(k) = Ez(k) - (dt/(eps0*dz)) * (Hy(k) - Hy(k-1));
    end

    % Absorbing boundary (right)
    Ez(Nz) = Ez(Nz-1);

    % Gaussian pulse injection
    Ez(source_pos) = Ez(source_pos) + ...
        exp(-((t(n)-t0)^2)/(tau^2)) * cos(omega * (t(n) - t0));

    Ez_history(n,:) = Ez;
    Hy_history(n,:) = Hy;
end

%% E-field at final time step
figure;
plot(z, Ez);
xlabel('Distance (m)');
ylabel('Ez (V/m)');
title('Gaussian Pulse — Electric Field at Final Time Step');

%% Velocity Analysis (envelope-based peak detection)
% Use points well within the domain to avoid source/boundary artifacts
k1 = 600;  k2 = 1200;

% Track the envelope peak using absolute value
[~, t1_idx] = max(abs(Ez_history(:,k1)));
[~, t2_idx] = max(abs(Ez_history(:,k2)));

% Ensure we detect forward-propagating peak (t2 > t1)
if t2_idx <= t1_idx
    % Fallback: search only after the pulse has passed k1
    [~, t2_idx] = max(abs(Ez_history(t1_idx+1:end, k2)));
    t2_idx = t2_idx + t1_idx;
end

delta_t  = (t2_idx - t1_idx) * dt;
distance = (k2 - k1) * dz;
v_sim    = distance / delta_t;

fprintf('\n--- Gaussian Pulse Velocity Analysis ---\n');
fprintf('Simulated velocity: %.2e m/s\n', v_sim);
fprintf('Theoretical c0    : %.2e m/s\n', c0);
fprintf('Relative error    : %.2f %%\n', abs(v_sim - c0)/c0 * 100);

%% Power vs Distance
n_plot  = 6000;
Ez_snap = Ez_history(n_plot,:);
Hy_snap = Hy_history(n_plot,:);

Hy_interp = [Hy_snap(1), (Hy_snap(1:end-1) + Hy_snap(2:end))/2];
Pz = Ez_snap .* Hy_interp;

figure;
plot(z, Pz);
xlabel('Distance (m)');
ylabel('Power Density (W/m^2)');
title('Gaussian Pulse — Poynting Vector vs Distance');
