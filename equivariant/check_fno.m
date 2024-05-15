clear;

load("fno_test.mat")

%restrict to a single batch and channel

b = 1;
c = 1;


data = squeeze( data (b,c,:,:) );
data2= squeeze( data2(b,c,:,:) );

tiledlayout(1,2);

nexttile
surf(data);
shading interp;

nexttile
surf(data2);
shading interp;
