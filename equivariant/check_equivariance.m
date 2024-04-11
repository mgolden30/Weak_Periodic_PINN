clear;
load("diff_rot.mat");

c = 1;
w1 = squeeze(o1(1,c,:,:));
w2 = squeeze(o2(1,c,:,:));

clf;
tiledlayout(1,3);

nexttile
imagesc(w1)
colorbar();
title(" symmetry then autoencoder")
axis square

nexttile
imagesc(w2)
colorbar();
title(" autoencoder then symmetry")
axis square

nexttile
diff = w1-w2;
%diff = fftshift(fft2(diff));
diff = abs(diff);
imagesc( diff )
title("difference");
axis square
colorbar();