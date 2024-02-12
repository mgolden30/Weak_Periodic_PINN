%{
Find Equilibria to forced Navier-Stokes
%}

N = 128;
[x,y] = meshgrid( (0:(N-1))/N*2*pi );

%guess an equilibria
forcing = 4*cos(4*y);
nu = 1/40;
w = sin(x) + cos(y);

%Macros for reshaping
to_vec = @(x) reshape(x, [N*N,1]);
to_mat = @(x) reshape(x, [N,N]  );

w = to_vec(w);

%%
clf;
load("candidate_5.mat");

maxit = 1024;
inner = 64;
outer = 1;
tol = 1e-6;

for i = 1:maxit
  %Define an objective function
  F = @(w) state_velocity( w, N, forcing, nu );
  F0= F(w);
  fprintf("step %03d: |F| = %e\n", i, norm(F0) );


  %Finite difference for the action of the Jacobian
  h = 1e-4;
  J = @(v) (F(w + h*v)-F0)/h;

  %Solve for 
  [dw, ~] = gmres( J, F0, inner, tol, outer);
  w = w - 0.1*dw;
  w = w - mean(w);

  imagesc( to_mat(w) );
  axis square
  colorbar();
  clim([-10 10]);
  colormap bluewhitered
  drawnow;
end
