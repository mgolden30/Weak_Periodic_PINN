clf

w = reshape( state(1:N*N), [N,N] );
w = abs(fft2(w))/N/N;

k = 0:N-1; k(k>N/2) = k(k>N/2) - N;
kx = k;
ky = k';

k_abs = sqrt(k.^2 + k'.^2);

to_vec = @(x) reshape(x, [numel(x),1] );

ms = 6;
scatter( to_vec(k_abs), to_vec(w), ms, 'filled' );
set(gca, 'yscale', 'log');
set(gca, 'xscale', 'log');

xlabel('k');
ylabel(['|\omega_{k}|']);

%Do a polynomial fit
idx = k_abs < 10 & k_abs > 4;
xs = k_abs(idx);
ys = w(idx);

pol = polyfit( log(to_vec(xs)), log(to_vec(ys)),1 );
hold on
  plot( 3:20, exp(polyval( pol, log(3:20))), "linewidth", 2, "color", "red" )
hold off

xx = [0.5 0.5];
yy = [0.6 0.8];
annotation('textarrow',xx,yy,'String','|\omega_k| \sim k^{-3.53}', "fontsize", 16, "color", "red")

set( gca, 'fontsize', 16);
