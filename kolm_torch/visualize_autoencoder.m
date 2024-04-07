clear;
load("equivariant_predictions.mat");

 vidObj = VideoWriter('autoenc.avi');
 open(vidObj);

clf
colormap jet

for tr= 1:size(w,2)
for t = 1:10:size(w,1)

  w_auto = squeeze( predictions(t,tr,:,:) );
  w_true = squeeze( w(t,tr,:,:) );
  latent = squeeze(latent_space( t,tr,:,:,1) );

  tiledlayout(2,2);
  
  nexttile
  nice_imagesc( w_true );
  title("DNS $\omega$", "interpreter", "latex");
  
  nexttile
  nice_imagesc( w_auto );
  title("autoencoder $\omega$", "interpreter", "latex");

  nexttile
  nice_imagesc( w_true - w_auto );
  title("difference", "interpreter", "latex");
  
  nexttile
  %plot( latent_space(i,:), 'o' );
  nice_imagesc( latent );
  clim([0,1]);
  title("latent space coordinates");

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