classdef ffnn_layer
   
   properties
        dW                  = [];
        W                   = [];
        M                   = [];
        MSE                 = 0;
        
        input               = [];
        weighted            = [];
        output              = [];
        deriv               = [];
        
        dims                = [];
        act_fn              = @bipolar_sigmoid;
        act_fn_deriv        = @bipolar_sigmoid_deriv;
        
        sens                = 1;    % activation temperature
        lr                  = 1e-3; % learning rate
        mw                  = 1e-3; % momentum weight
        mv                  = 1e-3; % momentum random variance
   end
    
   
   methods
       
       
       % @param     dims        [nOut,nIn]
       % @param     seed_var    weighting matrix seeding variance
       function layer       = ffnn_layer(dims,seed_var)           
           layer.M          = zeros(dims+[0,1]); 
           layer.W          = randn(dims+[0,1])*seed_var;
           
           layer.input      =  ones(dims(2)+1,1);
           layer.output     = zeros(dims(1)  ,1);
           
           layer.dims       = dims;
       end
       
              
       function layer       = activate(layer,input)
            layer.input(1:layer.dims(2),1) = input;
            layer.weighted  = layer.W*layer.input;
            layer.output    = layer.act_fn(layer.sens,layer.weighted);
       end
       
       
       function layer       = calc_reweight(layer,error)
            layer.MSE       = norm(error);
            layer.deriv     = layer.act_fn_deriv(layer.sens,layer.weighted,layer.output).*error;
            
            layer.dW        =-layer.lr*layer.deriv*transpose(layer.input);
            layer.W         = layer.W  + layer.dW + layer.mw*layer.M;
            layer.M         = layer.dW + layer.mv * randn(layer.dims(1),layer.dims(2)+1);
       end
       
       
       function layer       = set_reweight(layer)
            layer.W         = layer.W + layer.dW;
       end
       
       
       function err         = bp_error_hidden(layer)
            err             = transpose(layer.W(:,1:layer.dims(2)))*layer.deriv;
       end
       
       
       function err         = bp_error_output(layer,target)
            err             = layer.output-target;
       end
        
       
       function layer       = set_activation_type(layer,type)
            if      strcmpi(type,'BIPOLAR_SIGMOID')
                layer.act_fn       = @bipolar_sigmoid;
                layer.act_fn_deriv = @bipolar_sigmoid_deriv; 
            elseif  strcmpi(type,'UNIPOLAR_SIGMOID')
                layer.act_fn       = @unipolar_sigmoid;
                layer.act_fn_deriv = @unipolar_sigmoid_deriv; 
            elseif  strcmpi(type,'SATURATION')
                
            else
                error('FFNN : Unrecognize activation spec');
            end
       end
   end
   
end

%% BIPOLAR SIGMOID
function output = bipolar_sigmoid(sen,input)
    output = tanh(sen*input);
end
function output = bipolar_sigmoid_deriv(sen,input,output)
    output = sen*(ones(size(output))-output.^2);
end

%% UNIPOLAR SIGMOID
function output = unipolar_sigmoid(sen,input)
    tmp    = ones(size(input));
    output = tmp./(tmp+exp(-input));
end
function output = unipolar_sigmoid_deriv(sen,input,output)
    output =sen*output.*(ones(size(output))-output);
end