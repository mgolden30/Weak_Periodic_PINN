%{
Visualize the learned timesteps
%}

clear;
clf;


dt0 = load("dt_0.mat");
dt1 = load("dt_1.mat");
dt2 = load("dt_2.mat");

dt = [dt0.dt; dt1.dt; dt2.dt];
dt = reshape(dt, [numel(dt),1]);

nbins = round(numel(dt)/6);
histogram(dt, nbins);

xlabel("\Delta t");
ylabel("counts");
