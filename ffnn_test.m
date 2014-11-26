close all;
clear;
clc;

nn      = ffnn([1,50,1],0.1);
nn      = nn.set_lr(1e-2);
nn      = nn.set_mw(5e-2);

dt = 0.01;
t_span = 0:dt:(2*pi);

for itr = 1:1000
    for t = t_span
        nn          = nn.f_prop(t);
        nn          = nn.b_prop((cos(t)+1)/2);
    end
    
    data = zeros(numel(t_span),1);
    idx  = 0;
    for t = t_span
        idx         = idx + 1;
        nn          = nn.f_prop(t);
        data(idx,1) = nn.output;
    end
    
    cla
    plot(t_span,data,t_span,(cos(t_span)+1)/2)
    pause(1e-3);
end