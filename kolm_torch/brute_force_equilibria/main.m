%{
Find Equilibria to forced Navier-Stokes
%}

N = 128;
[x,y] = meshgrid( (0:(N-1))/N*2*pi );

%guess an equilibria
forcing = 4*cos(4*y);
nu = 1/40;
w = sin(x-1).*sin(y+2) + sin(2*x).*sin(2*y+3) - cos(x+4).*sin(3*y-7);
%w = -sin(3*x-2).*sin(y+x) + cos(x-y-1) + sin(y-1);
w = w - mean(w, 'all');
w = 10*w;

%Macros for reshaping
to_vec = @(x) reshape(x, [N*N,1]);
to_mat = @(x) reshape(x, [N,N]  );

u0 = -0.01;
state = [to_vec(w); u0];

%%
clf;
%load("ECS\EQ2.mat");

damp  = 0.1;
maxit = 128;
inner = 32;
outer = 1;
tol   = 1e-2; 

for i = 1:maxit
  %Define an objective function
  F = @(state) state_velocity( state, N, forcing, nu );
  F0= F(state);

  %Finite difference for the action of the Jacobian
  %h = 1e-4;
  %J = @(v) (F(w + h*v)-F0)/h;
  J = @(v) velocity_jacobian( state, v, N, forcing, nu );
  
  %Solve for
  X0 = zeros(N*N+1, 5);
  k=1;
  X0(1:N*N,1) = to_vec( cos(k*x) );
  X0(1:N*N,2) = to_vec( sin(k*x) );
  X0(1:N*N,3) = to_vec( cos(k*y) );
  X0(1:N*N,4) = to_vec( sin(k*y) );
  X0(N*N+1,5) = 1;

  %X0 = [];

  [ds, res] = block_gmres( J, F0, X0, tol, inner, outer);
  %[dw, ~] = gmres( J, F0, inner, tol, outer);
  state = state - damp*ds;
  %w = w - mean(w);
  fprintf("step %03d: |F| = %e\t res = %e\n", i, norm(F0), res );

  figure(1);
  make_nice_images( state, F0, N );
  drawnow;
end




function make_nice_images( state, F0, N )
  %tiledlayout(1,2);

  w = reshape( state(1:N*N), [N,N] );
  u0 = state(N*N+1);

  k = 0:N-1; k(k>N/2) = k(k>N/2) - N;
  kx = k;
  ky = k';

  %Construct spectral operators to go from vorticity to velocity
  k_sq = kx.^2 + ky.^2;
  to_u =  1i*ky./k_sq;
  to_v = -1i*kx./k_sq;
  to_u(1,1) = 0;
  to_v(1,1) = 0;

  wf = fft2(w);

  %plot every "ss"th velocity vector
  ss  = 4;
  u   = real(ifft2( to_u .* wf )) + u0;
  v   = real(ifft2( to_v .* wf ));
  [i,j] = meshgrid(1:N);
  turn_off = (mod(i,ss) ~=0) | (mod(j,ss)~=0);
  u(turn_off) = 0;
  v(turn_off) = 0;

 
  imagesc( w );
  hold on
  scale = 2;
  quiver( u, v, scale, "color", "black", "linewidth", 1)
  hold off
  set(gca, "ydir", "normal");
  axis square
  cb = colorbar();
  clim([-1 1]*10);
  set(cb, "XTick", [-10 0 10]);
  colormap bluewhitered;

  xticks([1, N]);
  xticklabels({'0','2\pi'})
  yticks([1, N]);
  yticklabels({'0','2\pi'})

  %title("$\nu = 1/40$");
  return;
end