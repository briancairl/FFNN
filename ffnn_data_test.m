try %#ok<TRYNC>
    delete(wb)
end
close all;
clear;
clc;
load('finalCS6923.mat')

M           = 200;
N           = numel(train_label);
perm        = randperm(N);
usevec      = perm(1:M);
keep        = var(train) > 1;
C           = nnz(keep);

train       = train(usevec,keep).';
train       = train./repmat(std(train),C,1);
train_label = 2*train_label(usevec)-3;


nn          = ffnn([C,100,1],0.1);

nn          = nn.set_lr(1e-8);
nn          = nn.set_mw(1e-3);
nn          = nn.set_mv(1e-20);
nn          = nn.set_sens(100);

wb          = waitbar(0,'Training...');
ITR         = 10000;

for itr = 1:ITR
    
    idx_set     = randperm(M);
    mse_total   = 0;
    for idx = idx_set
        
        nn = nn.f_prop(train(:,idx));
        
        nn = nn.b_prop(train_label(idx));
        
        mse_total = mse_total + nn.MSE;
    end
    
    cla
    hold on
    stem(train_label,'r');
    test_out = nn.run(train);
    stem( (test_out>0) - (test_out<0),'g');
    
    set(wb,'Name',['MSE : ',num2str(mse_total)]);
    waitbar(itr/ITR);
    
    pause(1e-9);
end
nn.save_net('TestNet')


delete(wb);