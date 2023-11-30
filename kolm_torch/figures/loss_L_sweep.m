%%Read the torch output

Ls = [8,16,32,64,128];%,16,32,64];

epoch = 2058;
clf();

for L = Ls
  load("../network_output/L_sweep/torch_output_L_" + L + "_epoch_" + epoch + ".mat");

  hold on
  plot( loss_history, 'linewidth', 2 );
  set(gca, 'yscale', 'log');
  hold off
end
legend({"L=8", "L=16", "L=32", "L=64", "L=128"} );

fs = 32;
ylabel("mean abs error", "FontSize", fs);
xlabel("epoch", "FontSize", fs );

xlim([1 epoch]);
xticks(xlim)
ylim([1e-3, 1e1]);
yticks(10.^[-3:1]);

set(gca,"fontsize", fs)