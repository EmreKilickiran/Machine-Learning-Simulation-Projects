
% task2_dielectric.m — Wave Propagation in Dielectric Media

% Part (a): Lossless dielectric (εr = 16, σ = 0)
%   - Validates reduced wave velocity: v = c0/√εr
% Part (b): Lossy medium (εr = 16, σ = 0.5 S/m)
%   - Validates exponential attenuation against analytical e^(-αz)


clear; clc; close all;

c0  = 3e8;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c0^2);

f     = 1e9;
omega = 2*pi*f;

%% PART A: Lossless Dielectric (εr = 16) 

fprintf('\nPart A: Lossless Dielectric\n');

eps_r_a = 16;
eps_a   = eps_r_a * eps0;
sigma_a = 0;

Nz_a = 1000;  dz_a = 4e-3;
z_a  = (0:Nz_a-1)*dz_a;
dt_a = dz_a/(2*c0);
Nt_a = 10000;

Ez_a = zeros(1,Nz_a);  Hy_a = zeros(1,Nz_a);
Ez_hist_a = zeros(Nt_a, Nz_a);
Hy_hist_a = zeros(Nt_a, Nz_a);
src_a = 2;

for n = 1:Nt_a
    for k = 1:Nz_a-1
        Hy_a(k) = Hy_a(k) - (dt_a/(mu0*dz_a))*(Ez_a(k+1)-Ez_a(k));
    end
    for k = 2:Nz_a-1
        coeff = dt_a*sigma_a/(2*eps_a);
        Ez_a(k) = (1-coeff)/(1+coeff)*Ez_a(k) ...
                - (dt_a/(eps_a*dz_a))/(1+coeff)*(Hy_a(k)-Hy_a(k-1));
    end
    Ez_a(Nz_a) = Ez_a(Nz_a-1);  % ABC
    Ez_a(src_a) = Ez_a(src_a) + sin(omega*n*dt_a);
    Ez_hist_a(n,:) = Ez_a;
    Hy_hist_a(n,:) = Hy_a;
end

% Velocity verification
k1=60; k2=160;
[~,t1]=max(Ez_hist_a(:,k1)); [~,t2]=max(Ez_hist_a(:,k2));
v_sim_a = (k2-k1)*dz_a / ((t2-t1)*dt_a);
v_theory_a = c0/sqrt(eps_r_a);

fprintf('Theoretical velocity: %.2e m/s\n', v_theory_a);
fprintf('Simulated velocity  : %.2e m/s\n', v_sim_a);
fprintf('Relative error      : %.2f %%\n', abs(v_sim_a-v_theory_a)/v_theory_a*100);

% Power in lossless dielectric
n_plot_a = Nt_a;
Ez_snap_a = Ez_hist_a(n_plot_a,:);
Hy_snap_a = Hy_hist_a(n_plot_a,:);
Hy_interp_a = [Hy_snap_a(1), (Hy_snap_a(1:end-1)+Hy_snap_a(2:end))/2];
Pz_a = Ez_snap_a .* Hy_interp_a;

figure;
plot(z_a, Pz_a);
xlabel('Distance (m)'); ylabel('Power Density (W/m^2)');
xlim([0 1]);
title('Power vs Distance — Lossless Dielectric (\epsilon_r = 16)');

%% PART B: Lossy Medium (σ = 0.5 S/m)

fprintf('\n Part B: Lossy Medium \n');

eps_r_b = 16;
eps_b   = eps_r_b * eps0;
sigma_b = 0.5;

Nz_b = 300;  dz_b = 1e-3;
z_b  = (0:Nz_b-1)*dz_b;
dt_b = dz_b/(2*c0);
Nt_b = 1200;

Ez_b = zeros(1,Nz_b);  Hy_b = zeros(1,Nz_b);
Ez_hist_b = zeros(Nt_b, Nz_b);
src_b = 2;

for n = 1:Nt_b
    for k = 1:Nz_b-1
        Hy_b(k) = Hy_b(k) - (dt_b/(mu0*dz_b))*(Ez_b(k+1)-Ez_b(k));
    end
    for k = 2:Nz_b-1
        coeff = dt_b*sigma_b/(2*eps_b);
        Ez_b(k) = (1-coeff)/(1+coeff)*Ez_b(k) ...
                - (dt_b/(eps_b*dz_b))/(1+coeff)*(Hy_b(k)-Hy_b(k-1));
    end
    Ez_b(Nz_b) = Ez_b(Nz_b-1);  % ABC
    Ez_b(src_b) = Ez_b(src_b) + sin(omega*n*dt_b);
    Ez_hist_b(n,:) = Ez_b;
end

% Power in lossy medium
n_plot_b = 900;
Ez_snap_b = Ez_hist_b(n_plot_b,:);
Hy_snap_b = Hy_b;  % Use current Hy state for approximate snapshot
Hy_interp_b = [Hy_snap_b(1), (Hy_snap_b(1:end-1)+Hy_snap_b(2:end))/2];
Pz_b = Ez_snap_b .* Hy_interp_b;

figure;
plot(z_b, Pz_b);
xlabel('Distance (m)'); ylabel('Power Density (W/m^2)');
title('Power vs Distance — Lossy Medium (\sigma = 0.5 S/m)');

% Attenuation comparison: simulated vs theoretical
alpha_th = sqrt(pi*f*mu0*sigma_b/2);
E0 = max(abs(Ez_snap_b));
Ez_theory = E0 * exp(-alpha_th * z_b);

figure;
plot(z_b, abs(Ez_snap_b), 'b', z_b, Ez_theory, 'r--', 'LineWidth', 1.2);
xlabel('Distance (m)'); ylabel('|Ez| (V/m)');
legend('Simulated', 'Theoretical e^{-\alpha z}');
title(sprintf('Field Attenuation (\\sigma = %.1f S/m)', sigma_b));
