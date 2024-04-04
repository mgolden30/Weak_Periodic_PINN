clear;
load("predictions.mat");

 vidObj = VideoWriter('autoenc.avi');
 open(vidObj);

clf
colormap jet

for i = 1:5:size(w_batch, 3)
  i
  w_auto = squeeze( predictions(i,:) );
  w_auto = reshape( w_auto, [64,64] );
  w_true = squeeze( w_batch(:,:,i) );


  tiledlayout(2,2);
  
  nexttile
  nice_imagesc( w_true' );
  title("DNS $\omega$", "interpreter", "latex");
  
  nexttile
  nice_imagesc( w_auto );
  title("autoencoder $\omega$", "interpreter", "latex");

  nexttile
  nice_imagesc( w_true' - w_auto );
  title("difference", "interpreter", "latex");
  
  nexttile
  plot( latent_space(i,:), 'o' );
  title("latent space coordinates");

  drawnow

  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end  
% Close the file.
close(vidObj);

function nice_imagesc( data )
  imagesc(data);
  colorbar();
  axis square
  clim( [-1, 1] );
end