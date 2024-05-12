%{
Visualize the learned timesteps
%}

clear;
clf;


k0 = load("kernel_0.mat");
k1 = load("kernel_1.mat");
k2 = load("kernel_2.mat");

k0 = k0.k;
k1 = k1.k;
k2 = k2.k;

tiledlayout(2,2);

for i = 0:1
    for j = 0:1
        nexttile
        imagesc( squeeze(k2(1,1,1 + 2*i+j,:,:)));
        axis square;
        set(gca, "ydir", "normal");
        xticks([]);
        yticks([]);
    end
end

num_weights = (prod(size(k0)) + prod(size(k1)) + prod(size(k2)) )/8