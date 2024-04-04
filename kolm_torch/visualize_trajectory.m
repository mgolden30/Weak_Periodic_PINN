clear;
load("w_traj.mat");


for i = 1:20:size(w,3)
  i
  imagesc( squeeze(w(:,:,i))' );
  colorbar();
  clim([-1, 1]*10);
  %colormap jet
  drawnow
  clf
end

