%   This function runs the BACF tracker on the video specified in "seq".
%   This function borrowed from SRDCF paper. 
%   details of some parameters are not presented in the paper, you can
%   refer to SRDCF/CCOT paper for more details.

function results = run_ARCF_H(seq, video_path, lr, gamma)
%   Default parameters used in the ICCV 2017 BACF paper

%   HOG feature parameters
hog_params.nDim   = 31;
params.video_path = video_path;
%   Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;

%   Global feature parameters 
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       = 0.019;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 50;           % number of Newton's iteration to maximize the detection scores
				% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0
%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.endFrame - seq.startFrame + 1;
params.seq_st_frame = seq.startFrame;
params.seq_en_frame = seq.endFrame;

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.admm_iterations = 2;
params.admm_lambda = 0.01;
params.admm_gamma = 0.71;

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default


%   Debug and visualization
params.visualization = 1;

%   Run the main function
results = tracker_HOG(params);
