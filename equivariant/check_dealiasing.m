%{
This script I check my greatest fear: advection is not equivariant unless
there is no aliasing possible
%}

n = 256;
w = randn( [n,n] );
w = w - mean(w,"all");

dx = 0.55;
dy = 0.21;
dt = 1;

nyq = n/2;
errs = zeros(nyq,1);
for k_max = 1:nyq
  w0 = dealias(w, k_max);
  imagesc(w0); drawnow;

  term1 = shift( advection( w0, dt ), dx, dy );
  term2 =  advection( shift(w0, dx, dy ), dt );

  errs(k_max) = mean(abs(term1 - term2), "all");
end

%%
scatter(1:nyq, errs);
title("|advec(shift) - shift(advec)|");
xlabel("dealias cutoff");
ylabel("error")
xlim([0, nyq]);
xticks([0, nyq/2, nyq])
set(gca,"yscale", "log");
function w = advection(w, dt)
  n = size(w,1);
  k = 0:n-1; k(k>n/2) = k(k>n/2) - n;

  to_u = 1i*k'./( k.^2 + k'.^2);
  to_v =-1i*k ./( k.^2 + k'.^2);
  to_u(1,1) = 0;
  to_v(1,1) = 0;
  
  wf = fft2(w);
  u = real(ifft2(to_u.*wf));
  v = real(ifft2(to_v.*wf));
  wx= real(ifft2(1i*k.*wf));
  wy= real(ifft2(1i*k.'.*wf));

  w = w - dt*(u.*wx + v.*wy);
end



function w = dealias(w, n_max)
  n = size(w,1);
  k = 0:n-1; k(k>n/2) = k(k>n/2) - n;

  %mask = k.^2 + k'.^2 >= n_max.^2;

  mask1 = abs(k)  >= n_max;
  mask2 = abs(k') >= n_max;
  mask = mask1 | mask2;

  wf = fft2(w);
  wf(mask) = 0;
  w  = real(ifft2(wf));
end


function w = shift(w, dx, dy)
  n = size(w,1);
  k = 0:n-1; k(k>n/2) = k(k>n/2) - n;

  wf = fft2(w);
  wf = wf .* exp(1i * k'*dx) .* exp(1i*k*dy);
  w  = real(ifft2(wf));


end