%%Write out a state to a name of your choosing
name = "REQ1";
save("ECS/" + name + ".mat", "state", "nu", "N", "forcing");
saveas(gcf, "ECS/pictures/" + name + ".png");