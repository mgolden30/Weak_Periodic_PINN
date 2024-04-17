clear;
load("w_traj.mat");

figure(1);
clf;
colormap jet


n=64;
k = 0:n-1; k(k>n/2) = k(k>n/2) - n;

mask = abs(k) >= 4;

w2 = fft2(w);
w2(mask, :,:,:) = 0;
w2(:,mask,:,:)  = 0;
w2 = real(ifft2(w2));



MSE = mean( (w - w2).^2, 'all' )
for tr= 1:size(w,4)
for t = 1:size(w,3)
  
  tiledlayout(1,3);

  nexttile
  imagesc( squeeze(w(:,:,t,tr))' );
  colorbar();
  clim([-1, 1]*10);
  %colormap jet
  title(tr)
  axis square

  
  nexttile
  imagesc( squeeze(w2(:,:,t,tr))' );
  colorbar();
  clim([-1, 1]*10);
  %colormap jet
  title(tr)
  axis square


    nexttile
  imagesc( squeeze(w2(:,:,t,tr))' - squeeze(w(:,:,t,tr))' );
  colorbar();
  %clim([-1, 1]*10);
  %colormap jet
  title(tr)
  axis square

  
  drawnow
  clf
end
end

