% =========================================================================
% task5_bragg_reflector.m — Periodic Bragg Reflector Structure
% =========================================================================
% Implements a multi-layer dielectric Bragg reflector with alternating
% permittivities (εA=2, εB=6), each layer λ/4 thick at 1 GHz.
% Constructive interference creates strong reflection at the design freq.
% =========================================================================

clear; clc; close all;

c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c0^2);

Nz = 800;  dz = 0.5e-3;
z  = (0:Nz-1)*dz;
dt = dz/(2*c0);
Nt = 2000;

%% Bragg Structure Parameters
f0      = 1e9;
omega   = 2*pi*f0;
lambda0 = c0/f0;

eps_r_A   = 2;     eps_r_B = 6;
N_periods = 5;

d_A = lambda0 / (4*sqrt(eps_r_A));
d_B = lambda0 / (4*sqrt(eps_r_B));
pts_A = round(d_A/dz);
pts_B = round(d_B/dz);

fprintf('Layer A: εr=%.0f, d=%.4f m (%d pts)\n', eps_r_A, d_A, pts_A);
fprintf('Layer B: εr=%.0f, d=%.4f m (%d pts)\n', eps_r_B, d_B, pts_B);
fprintf('Periods: %d\n', N_periods);

%% Build permittivity profile
eps_r = ones(1, Nz);
start_idx = 200;

for p = 0:N_periods-1
    a_start = start_idx + p*(pts_A + pts_B);
    a_end   = a_start + pts_A - 1;
    b_start = a_end + 1;
    b_end   = b_start + pts_B - 1;
    if b_end > Nz, break; end
    eps_r(a_start:a_end) = eps_r_A;
    eps_r(b_start:b_end) = eps_r_B;
end

eps   = eps_r * eps0;
sigma = zeros(1, Nz);

%% FDTD Simulation
Ez = zeros(1,Nz);  Hy = zeros(1,Nz);
Ez_history = zeros(Nt, Nz);
Hy_history = zeros(Nt, Nz);
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
    Ez(Nz) = Ez(Nz-1);  % ABC
    Ez(source_pos) = Ez(source_pos) + sin(omega*n*dt);
    Ez_history(n,:) = Ez;
    Hy_history(n,:) = Hy;
end

%% E-field at final time step
figure;
plot(z, Ez);
xlabel('Distance (m)'); ylabel('Ez (V/m)');
title('Ez — Bragg Reflector (5 periods, 1 GHz)');

%% Power (Poynting vector) vs Distance
Ez_snap = Ez_history(Nt,:);
Hy_snap = Hy_history(Nt,:);
Hy_interp = [Hy_snap(1), (Hy_snap(1:end-1)+Hy_snap(2:end))/2];
Pz = Ez_snap .* Hy_interp;

figure;
plot(z, Pz);
xlabel('Distance (m)'); ylabel('Power Density (W/m^2)');
title('Poynting Vector vs Distance — Bragg Reflector');
