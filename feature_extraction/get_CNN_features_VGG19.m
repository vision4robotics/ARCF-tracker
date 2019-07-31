% GET_FEATURES: Extracting hierachical convolutional features

% function feat = get_CNN_features_VGG19(im, cos_window, layers)
function feat = get_CNN_features_VGG19(im, fparams, gparams)

global net
% global enableGPU
enableGPU = gparams.use_gpu;

if isempty(net)
    initial_net();
end

sz_window = size(fparams.cos_window);

% Preprocessing
img = single(im);        % note: [0, 255] range
img = imResample(img, net.meta.normalization.imageSize(1:2));

average=net.meta.normalization.averageImage;

if numel(average)==3
    average=reshape(average,1,1,3);
end

img = bsxfun(@minus, img, average);

if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);

% Initialize feature maps
feat = cell(length(fparams.layers), 1);

for ii = 1:length(fparams.layers)
    
    % Resize to sz_window
    if enableGPU
        x = gather(res(fparams.layers(ii)).x); 
    else
        x = res(fparams.layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));
    
    % windowing technique
    if ~isempty(fparams.cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;
end

end
