%%Read the torch output

load("network_output2/torch_output_4000.mat");
fs = 20;

clf()
semilogy( loss_history, 'linewidth', 2, 'color', 'black');
ylabel("MSE loss", "fontsize", fs);
xlabel("epoch", "fontsize", fs);
drawnow;
saveas( gcf, "loss_histroy.png");
return


w = f(:,:,:,1);
u = f(:,:,:,2);
v = f(:,:,:,3);
T = f(:,:,:,4);
a = f(:,:,:,5);

T(1,1,1)
a(1,1,1)

N = size(w,1);

vidObj = VideoWriter('RPO.mp4');
vidObj.FrameRate = 4;
open(vidObj);
 
M = size(w,3);
for t= 1:M
  imagesc(w(:,:,t).');
  set(gca, 'ydir', 'normal');
  axis square;
  
  title( "frame=" + t );

  colorbar();
  clim([-5 5]);
  colormap bluewhitered

  drawnow;
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
% Close the file.
close(vidObj);
