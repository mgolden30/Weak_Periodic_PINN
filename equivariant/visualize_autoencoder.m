clear;
load("equivariant_predictions.mat");

%{
mean_w = mean(predictions, [3,4]);
imagesc(mean_w)
colorbar();
return
%}

 vidObj = VideoWriter('autoenc.avi');
 open(vidObj);

clf
colormap jet

for tr= 1:size(w,2)
for t = 1:1:size(w,1)

  w_auto = squeeze( predictions(t,tr,:,:) );
  w_true = squeeze( w(t,tr,:,:) );

  latent = squeeze(latent_space( t,tr,:,:) );

  tiledlayout(1,3);
  
  nexttile
  %w_true = log10(abs(fftshift(fft2(w_true))));
  nice_imagesc( w_true );
  %clim([-10, -2])
  title("DNS $\omega$", "interpreter", "latex");
  
  nexttile
  %latent = abs(fftshift(fft2(latent))) 
  nice_imagesc( latent );
  clim([-1,1]*1e2);
  title("latent space coordinates");


  nexttile
  %w_auto = log10(fftshift(abs(fft2(w_auto))));
  nice_imagesc( w_auto );
  %clim([-10, -2])
  title("autoencoder $\omega$", "interpreter", "latex");

  %{
  nexttile
  nice_imagesc( w_true - w_auto );
  title("difference", "interpreter", "latex");
  %}

  set(gcf, "color", "w");

  drawnow

  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
end
% Close the file.
close(vidObj);

function nice_imagesc( data )
  imagesc(data');
  colorbar();
  axis square
  clim( [-1, 1] * 10 );

  set(gca, 'ydir', 'normal');
  
end