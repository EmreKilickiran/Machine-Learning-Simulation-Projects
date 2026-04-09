% =========================================================================
% task3_interface.m — Single Dielectric Interface (Reflection/Transmission)
% =========================================================================
% Domain split: left (εr=1, free space) | right (εr=4, dielectric)
% Validates reflection coefficient Γ and transmission coefficient T
% against analytical Fresnel equations.
% =========================================================================

clear; clc; close all;

c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c0^2);

Nz = 400;  dz = 1e-3;
z  = (0:Nz-1)*dz;
dt = dz/(2*c0);
Nt = 1500;

%% Material: two-region domain
eps_r1 = 1;   eps_r2 = 4;
eps_r  = [eps_r1*ones(1, Nz/2), eps_r2*ones(1, Nz/2)];
eps    = eps_r * eps0;
sigma  = zeros(1, Nz);

Ez = zeros(1,Nz);  Hy = zeros(1,Nz);
Ez_history = zeros(Nt, Nz);

f = 1e9;  omega = 2*pi*f;  source_pos = 20;

%% FDTD Loop
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
end

%% E-field at final time step
figure;
plot(z, Ez);
xlabel('Distance (m)'); ylabel('Ez (V/m)');
title('Electric Field at Final Time Step (Interface in Middle)');
xline(z(Nz/2), '--r', 'Interface', 'LabelVerticalAlignment','bottom');

%% Theoretical reflection/transmission coefficients
eta1  = sqrt(mu0 / (eps_r1*eps0));
eta2  = sqrt(mu0 / (eps_r2*eps0));
Gamma = (eta2 - eta1) / (eta2 + eta1);
T     = 2*eta2 / (eta2 + eta1);

fprintf('\n--- Reflection Analysis ---\n');
fprintf('Theoretical Gamma: %.3f\n', Gamma);
fprintf('Theoretical T    : %.3f\n', T);
fprintf('Reflectance |Γ|² : %.3f\n', Gamma^2);
