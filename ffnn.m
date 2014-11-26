classdef ffnn
    properties
        layer_spec      = [];
        layers          = {};
        n               = 0;
        output          = [];
    end

    methods
        function nn     = ffnn(varargin)
            if      nargin == 1
                nn = nn.load_net(varargin{1});
            elseif  nargin == 2
                nn.n            = numel(varargin{1})-1;
                nn.layers       = cell(nn.n,1);
                nn.layer_spec   = varargin{1};

                for idx = 1:nn.n
                    size        = varargin{1}((idx+1):-1:(idx));
                    nn.layers{idx} = ffnn_layer(size,varargin{2});
                end
            else
                error('FFNN : Bad argument count');
            end
        end
        
        
        function save_net(nn,fname)
            save(fname,'nn');
        end
        
        
        function nn = load_net(nn,fname)
            ld = load(fname);
            nn = ld.nn; 
        end
        
        
        function nn     = f_prop(nn,input)
            nn.layers{1}= nn.layers{1}.activate(input);
            for idx = 2:nn.n
                nn.layers{idx} = nn.layers{idx}.activate(nn.layers{idx-1}.output);
            end
            nn.output = nn.layers{nn.n}.output;
        end
         
        
        function nn     = b_prop(nn,target)
            error       = nn.layers{nn.n}.bp_error_output(target);
            for idx = nn.n:-1:1
                nn.layers{idx} = nn.layers{idx}.calc_reweight(error);
                error          = nn.layers{idx}.bp_error_hidden();
            end
            for idx = 1:nn.n
                nn.layers{idx} = nn.layers{idx}.set_reweight();
            end
        end
        
        
        function nn     = set_lr(nn,lr)
            for idx = 1:nn.n
                nn.layers{idx}.lr = lr;
            end
        end
        
    
        function nn     = set_mw(nn,mw)
            for idx = 1:nn.n
                nn.layers{idx}.mw = mw;
            end
        end
   
        
        function nn     = set_mv(nn,mv)
            for idx = 1:nn.n
                nn.layers{idx}.mv = mv;
            end
        end
        
        
        function nn     = set_sens(nn,sens)
            for idx = 1:nn.n
                nn.layers{idx}.sens = sens;
            end
        end
        
        
        
        function nn     = set_activaton_type(nn,idx,type)
            nn.layers{idx} = nn.layers{idx}.set_activation_type(type);
        end
        
        
        function Y      = run(nn,X)
            N = numel(X(1,:));
            Y = zeros(nn.layers{nn.n}.dims(1),N);
            for idx = 1:N
               nn = nn.f_prop(X(:,idx));
               Y(:,idx) = nn.output;
            end
        end
        
        
        function mse    = MSE(nn)
            mse = 0;
            for idx = 1:nn.n
                mse = mse + nn.layers{idx}.MSE;
            end
        end
    end
end