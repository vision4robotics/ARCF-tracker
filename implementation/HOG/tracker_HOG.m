% This function implements the BACF tracker.

function [results] = tracker_HOG(params)

%   Setting parameters for local use.
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;
admm_gamma          = params.admm_gamma;
admm_iterations     = params.admm_iterations;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

% Define data types
if params.use_gpu
    params.t_global.data_type = zeros(1, 'single', 'gpuArray');
else
    params.t_global.data_type = zeros(1, 'single');
end

%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% Calculate feature dimension
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;

% allocate memory for multi-scale tracking
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
small_filter_sz = floor(base_target_sz/featureRatio);

% initialize previous response map
M_prev = zeros(use_sz);

loop_frame = 1;
for frame = 1:numel(s_frames)
    %load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    tic();
    
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    if frame > 1
        for scale_ind = 1:nScales
            multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
        end
        xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window));
        responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
        
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        if interpolate_response == 2
            % use dynamic interp size
            interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
        end
        responsef_padded = resizeDFT2(responsef, interp_sz);
        
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        
        % find maximum peak
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        end

        % calculate translation
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
        end
        
        % set the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position
        old_pos = pos;
        pos = pos + translation_vec;
        
        % find peak in the map
        M_curr = fftshift(response(:,:,sind));
        max_M_curr = max(M_curr(:));
        [id_ymax_curr, id_xmax_curr] = find(M_curr == max_M_curr);
        
        % do shifting of previous response map
        shift_y = id_ymax_curr - id_ymax_prev;
        shift_x = id_xmax_curr - id_xmax_prev;
        sz_shift_y = size(shift_y);
        sz_shift_x = size(shift_x);
        if(sz_shift_y(1) > 1)
            shift_y = shift_y(1);
        end
        if(sz_shift_x(1) > 1)
            shift_x = shift_x(1);
        end

        M_prev = circshift(M_prev,shift_y,1);
        M_prev = circshift(M_prev,shift_x,2);

        % map difference
        map_diff(frame) = norm(abs(M_prev - M_curr));
    end
        
    % extract training sample image region
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
    
    % extract features and do windowing
    xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
    
    if (frame == 1)
        model_xf = xf;
    else
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
    end
    
    g_f = single(zeros(size(xf)));
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
    
    T = prod(use_sz);
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    %   ADMM
    while (i <= admm_iterations)

        A = mu / (admm_gamma + 1);
        B = S_xx + T * A;
        S_lx = sum(conj(model_xf) .* l_f, 3);
        S_hx = sum(conj(model_xf) .* h_f, 3);
        g_f = (1 / (1 + admm_gamma)) * ( (((1/(T*A)) * bsxfun(@times, yf, model_xf)) + (admm_gamma / A) * bsxfun(@times, M_prev, model_xf) - ((1/A) * l_f) + (mu/A)* h_f) - ...
            bsxfun(@rdivide,(((1/(T*A)) * bsxfun(@times, model_xf, (S_xx .* yf))) + (admm_gamma/A) * bsxfun(@times, model_xf, (S_xx .* M_prev)) - ((1/A) * bsxfun(@times, model_xf, S_lx)) + (mu/A)*(bsxfun(@times, model_xf, S_hx))), B));        
        
        %   solve for H
        h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end
    
    % save current response map
    if(frame == 1)
        responsef = permute(sum(bsxfun(@times, conj(g_f), xf), 3), [1 2 4 3]);
        responsef_padded = resizeDFT2(responsef, interp_sz);
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        M_prev = fftshift(response);
        max_M_prev = max(M_prev(:));
        [id_ymax_prev,id_xmax_prev]=find(M_prev == max_M_prev);
    else
        M_prev = M_curr;
        max_M_prev = max_M_curr;
        id_ymax_prev = id_ymax_curr;
        id_xmax_prev = id_xmax_curr;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
            alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 100, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            
            hold off;
        end
        drawnow
    end
    loop_frame = loop_frame + 1;
end
close all

%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
results.map_diff = map_diff;
