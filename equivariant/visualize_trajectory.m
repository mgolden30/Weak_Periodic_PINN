clear;
clf;

load("w_traj.mat");


batch = size(w,1);
nt = size(w,2);

% sample and transpose to plot correclt in MATLAB
samp = @(w,b,t) squeeze(w(b,t,:,:))';


figure(1);
clf;

for b = 1:batch
for t = 1:1%:nt
  w0 = samp(w,b,t);
  imagesc( w0 );
  colorbar();
  clim([-1, 1]*10);
  title("batch number = " + b);
  axis square

  drawnow
end
end

