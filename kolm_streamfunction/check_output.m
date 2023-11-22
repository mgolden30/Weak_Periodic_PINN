load('C:/Users/wowne/NN_search/output_1.mat');

T(1,1,1)
a(1,1,1)

for t = 1:size(w,3)
  clf;
  
  imagesc( w(:,:,t) );
  set(gca, 'ydir', 'normal');

  colormap bluewhitered
  clim([ -5 5] )
  colorbar();
  
  %{
  hold on
    scale = 1;
    quiver( u, v, scale, 'color', 'black', 'LineWidth', 1 );
  hold off
  %}
  
  axis square

  title("t=" + (t-1)/size(w,3) * T(1,1,1) );

  drawnow
end