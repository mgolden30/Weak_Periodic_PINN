% Test subsampling
clear;

load( "subsample.mat" );

n = 64;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;

k2 = k(1:n/2+1);

k
k2

%try just default subsampling
idx1 = abs(k) <= n/4;
idx1( k == -n/4) = 0; %kill negative nyquist?
idx2 = abs(k2) <= n/4;

ss = cl( idx1, idx2 );

tiledlayout(2,2)

nexttile
imagesc( log10(abs(cl)) );
colorbar();

nexttile
imagesc( log10(abs(cs)) );
colorbar();

nexttile
imagesc( log10(abs(ss)) );
colorbar();

nexttile
diff = ss-4*cs;
%diff = cs-cs2;
imagesc( log10( abs(diff) ) );
colorbar();
title("diff")