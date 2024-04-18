clear;

str = "diff_trans"

load( str  + ".mat");
out_name = str + ".png";

c = 1;
w1 = squeeze(o1(1,c,:,:));
w2 = squeeze(o2(1,c,:,:));

clf;
tiledlayout(1,3);

nexttile
nice_imagesc(w1)
colorbar();
title(" symmetry then autoencoder")
axis square

nexttile
nice_imagesc(w2)
colorbar();
title(" autoencoder then symmetry")
axis square

nexttile
diff = w1-w2;
%diff = fftshift(fft2(diff));
diff = abs(diff);
nice_imagesc( diff )
title(sprintf("difference: max(abs(diff)) = %.2e", max(max(abs(diff)))));
axis square
colorbar();

drawnow
saveas(gcf, out_name)

function nice_imagesc( data )
  imagesc(data');
  colorbar();
  axis square
  clim( [-1, 1] * 10 );

  set(gca, 'ydir', 'normal');
  
end