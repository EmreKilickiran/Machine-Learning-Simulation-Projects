
% task1_sinusoidal.m — Free Space Wave Propagation (Sinusoidal Source)

% Simulates 1D EM wave propagation in free space using the FDTD method
% with the Yee algorithm. A 1 GHz sinusoidal source excites the domain.

% Validates: wave velocity (< 3% error vs analytical c0)


clear; clc; close all;

%% Physical Constants
c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0 * c0^2);

%% Grid & Time Parameters
Nz = 3000;
dz = 0.5e-3;
z  = (0:Nz-1)*dz;
dt = dz/(2*c0);              % Courant stability: dt <= dz/c0
Nt = 20000;

%% Field Initialization
Ez = zeros(1, Nz);
Hy = zeros(1, Nz);
Ez_history = zeros(Nt, Nz);
Hy_history = zeros(Nt, Nz);

%% Source Parameters
f     = 1e9;
omega = 2*pi*f;
source_pos = 2;

%% FDTD Time-Marching Loop
for n = 1:Nt
    % Update Hy (Yee: Hy is offset by half-step in space and time)
    for k = 1:Nz-1
        Hy(k) = Hy(k) - (dt/(mu0*dz)) * (Ez(k+1) - Ez(k));
    end

    % Update Ez
    for k = 2:Nz-1
        Ez(k) = Ez(k) - (dt/(eps0*dz)) * (Hy(k) - Hy(k-1));
    end

    % First-order absorbing boundary condition (right boundary)
    Ez(Nz) = Ez(Nz-1);

    % Soft sinusoidal source injection
    Ez(source_pos) = Ez(source_pos) + sin(omega * n * dt);

    % Store fields
    Ez_history(n,:) = Ez;
    Hy_history(n,:) = Hy;
end

%% Visualization: E-field snapshots
snapshot_steps = round(linspace(Nt*0.1, Nt, 6));
figure;
for i = 1:6
    subplot(2,3,i);
    plot(z, Ez_history(snapshot_steps(i),:), 'b');
    ylim([-0.3 0.3]);
    xlabel('Distance (m)');
    ylabel('Ez (V/m)');
    title(sprintf('Time step: %d', snapshot_steps(i)));
end
sgtitle('Sinusoidal Wave Propagation in Free Space');

%% Velocity Analysis
k1 = 50;   k2 = 150;
[~, t1_idx] = max(Ez_history(:,k1));
[~, t2_idx] = max(Ez_history(:,k2));

delta_t  = (t2_idx - t1_idx) * dt;
distance = (k2 - k1) * dz;
v_sim    = distance / delta_t;

fprintf('\n--- Wave Velocity Analysis ---\n');
fprintf('Simulated velocity: %.2e m/s\n', v_sim);
fprintf('Theoretical c0    : %.2e m/s\n', c0);
fprintf('Relative error    : %.2f %%\n', abs(v_sim - c0)/c0 * 100);

%% Power (Poynting Vector) vs Distance
n_plot  = 6000;
Ez_snap = Ez_history(n_plot,:);
Hy_snap = Hy_history(n_plot,:);

% Poynting vector: S = E x H (interpolate Hy to Ez grid points)
Hy_interp = [Hy_snap(1), (Hy_snap(1:end-1) + Hy_snap(2:end))/2];
Pz = Ez_snap .* Hy_interp;

figure;
plot(z, Pz);
xlabel('Distance (m)');
ylabel('Power Density (W/m^2)');
title(sprintf('Poynting Vector vs Distance (t = %d)', n_plot));
