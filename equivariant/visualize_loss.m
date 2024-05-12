%{
Plot trainnig and test errors
%}

clear;
clf;

files = {"loss.mat", "loss.mat"};
colors = {"black", "blue"};
hold on

for i = 1:numel(files)
load(files{i});

ep = numel(train_loss);

train_loss = train_loss / train_loss(1);
test_loss  = test_loss  / test_loss(1);

lw = 2; %width

plot( 1:ep, train_loss, 'linewidth', lw, 'color', colors{i}, 'linestyle', '-');
plot( 1:ep, test_loss,  'linewidth', lw, 'color', colors{i}, 'linestyle', '--');

end

set(gca, 'xscale', 'log' );
set(gca, 'yscale', 'log' );

legend( {"Train", "Test"} );
xlabel("epoch");
ylabel("normalized MSE");

xlim([1, ep]);
xticks(2.^[1:10] );

set(gcf, 'color', 'w');

fprintf("done\n");