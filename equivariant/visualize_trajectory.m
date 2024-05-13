clear;
clf;

load("w_traj.mat");


mode = "movie"; %"movie" or "snapshots"
%mode = "snapshots";

colormap jet

batch = size(w,1);
nt = size(w,2);


plot_statistics( w );

% sample and transpose to plot correctly in MATLAB
samp = @(w,b,t) squeeze(w(b,t,:,:))';

if mode == "movie"

for b = 1:batch
for t = 1:nt
  figure(1);

  w0 = samp(w,b,t);
  imagesc( w0 );
  colorbar();
  clim([-1, 1]*10);
  title("batch number = " + b);
  axis square

  drawnow
end
end

end

if mode == "snapshots"
  nn = ceil( sqrt(batch) );

  tiledlayout(nn,nn);

  for b = 1:nn*nn
    nexttile
    t = 1; %plot the first timestep
    w0 = samp(w,b,t);
    imagesc( w0 );
    colorbar();
    clim([-1, 1]*10);
    title("batch number = " + b);
    axis square

  end

end

function plot_statistics( w )
  figure(2);

  histogram( mean(w.^2, [3,4]), "NumBins", 100);
  title("Enstrophy");

  drawnow;
end