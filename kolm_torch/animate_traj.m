clear;

load("../kolm.mat");

for t = 1:size(w,3)
   figure(1);
   nice_plot( w(:,:,t) );
   title(t)
   drawnow
end

%%

t0 = 128; %transient time to start after

r1 = 2048; %basepoints
r2 = 256; %look this far into future

d = zeros(r2,r1);

to_vec = @(x) reshape( x, [numel(x),1]);

N = size(w,1);

for i = 1:r1
    for j = 1:r2
        d(j,i) = norm( to_vec( psi(:,:,t0+i)) - to_vec( psi(:,:,t0+i+j)) )/N;
    end
end

%%
figure(1);
imagesc(d)
colorbar();
clim([0 1]);
colormap jet
set(gca,'ydir','normal')
pbaspect([r1/r2 1 1]);

%%

%Pick a set of points
i = 800;
j = 100;

hold on
scatter( i,j, 'filled' );
hold off

figure(2);

tiledlayout(2,2);

nexttile
nice_plot( w(:,:,t0 + i) );
title("initial")

nexttile
nice_plot( w(:,:,t0 + i + j) );
title("final")

nexttile
enst = mean(w(1,1,t0 + (i:i+j)).^2, [1,2]);
plot( squeeze(enst) );
title('enstrophy')
xlabel('timestep');

nexttile
for t = 1:j
  nice_plot( w(1:8:end,1:8:end, t0 + i + t ));
  title(t);
  drawnow
end

function nice_plot(w)
  imagesc( w);
  axis square

  colorbar();
  clim([-5 5])
  colormap bluewhitered

  set(gca, 'ydir', 'normal');
end