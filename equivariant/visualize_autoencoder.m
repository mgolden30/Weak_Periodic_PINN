clear;
load("equivariant_predictions.mat");

%{
mean_w = mean(predictions, [3,4]);
imagesc(mean_w)
colorbar();
return
%}

vidObj = VideoWriter('autoenc.avi');
vidObj.FrameRate = 10;
open(vidObj);

clf
colormap jet

mean_true = mean( w.^2,[3,4] );
mean_auto = mean( predictions, [3,4] );


%{
clf
imagesc(mean_auto)
colorbar();
return
%}

%generate a colormap that is centered at zero
clim([-1,1]);
colormap bluewhitered

for tr= 1:size(w,2)
for t = 1:1:size(w,1)
  
  %Take this particular frame
  w_auto = squeeze( predictions(t,tr,:,:) );
  w_true = squeeze( w(t,tr,:,:) );

  %The latent space is more complicated
  latent = squeeze(latent_space( t,tr,:,:,:) );
  num_channels = size(latent,1);
  l = [];
  for i = 1:num_channels
    l = [l, squeeze(latent(i,:,:))];
  end

  tiledlayout(2,2);
  
  nexttile
  %w_true = log10(abs(fftshift(fft2(w_true))));
  nice_imagesc( w_true );
  %clim([-10, -2])
  title("DNS $\omega$", "interpreter", "latex");
  

  nexttile
  %w_auto = log10(fftshift(abs(fft2(w_auto))));
  nice_imagesc( w_auto );
  %clim([-10, -2])
  rel_err = norm(w_true - w_auto)/norm(w_true);
  title_str = sprintf( "autoencoder $\\omega$, relative error = %.3f", rel_err);
  title(title_str, "interpreter", "latex");


  nexttile
  %latent = abs(fftshift(fft2(latent))) 
  nice_imagesc( l );
  clim([-1,1]*20);
  title("latent space coordinates");
  pbaspect([1,2,1])
  yline(8.5, "linewidth", 3);

  
  
  nexttile
  nice_imagesc( w_true - w_auto );
  title_str = sprintf( "difference relative error = %.3f", rel_err);
  title("difference", "interpreter", "latex");
  


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