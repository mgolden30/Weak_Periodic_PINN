clear;
load("w_traj.mat");

figure(1);
clf;
colormap jet


for tr= 1:size(w,4)
for t = 1:size(w,3)
  t
  imagesc( squeeze(w(:,:,t,tr))' );
  colorbar();
  clim([-1, 1]*10);
  %colormap jet
  title(tr)
  drawnow
  clf
end
end

