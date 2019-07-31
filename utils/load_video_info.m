function [seq, ground_truth] = load_video_info(video_path,video,ground_truth_base_path)

    seqs=configSeqs;
    seq.video_path = strcat(video_path, video);
    i=1;
    while ~strcmpi(seqs{i}.name,video)
            i=i+1;
    end
    
    seq.VidName = seqs{i}.name;
    seq.st_frame = seqs{i}.startFrame;
    seq.en_frame = seqs{i}.endFrame;
    
    seq.ground_truth_fileName = seqs{i}.name;
    ground_truth = dlmread([ground_truth_base_path seq.ground_truth_fileName '.txt']);


%ground_truth = dlmread([video_path '/groundtruth_rect.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);
seq.ground_truth = ground_truth;

img_path = [video_path video '\'];
%img_path = [video_path '/img/'];

if exist([img_path num2str(1, '%06i.png')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.png']);
elseif exist([img_path num2str(1, '%06i.jpg')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.jpg']);
elseif exist([img_path num2str(1, '%06i.bmp')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%06i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);

end

