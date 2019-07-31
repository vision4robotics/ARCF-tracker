% Loads relevant information of UAV123 in the given path.
% Fuling Lin, 20190101

function seq = load_video_info_UAV123(video_name, database_folder, ground_truth_path, type)

    switch type
        case 'UAV123_10fps'
            seqs = configSeqs(database_folder, type);       % database_folder是包含了所有数据集的文件夹
        case 'UAV123'
            seqs = configSeqs(database_folder, type);       % type: UAV123_10fps, UAV123, UAV123_20L
        case 'UAV123_20L'
            seqs = configSeqs(database_folder, type);
    end
    
    i=1;
    while ~strcmpi(seqs{i}.name,video_name) % 为了获得configSeqs中对所选择数据集的设置
            i=i+1;
    end
    
    seq.video_name = seqs{i}.name;          % 获得数据集名称，与video_name相同
    seq.name = seqs{i}.name;
    seq.video_path = seqs{i}.path;          % 数据集所在路径，包含图片序列的文件夹
    seq.st_frame = seqs{i}.startFrame;      % 开始帧数
    seq.en_frame = seqs{i}.endFrame;        % 结束帧数
    seq.len = seq.en_frame-seq.st_frame+1;  % 序列长度
    
    ground_truth = dlmread([ground_truth_path '\' seq.video_name '.txt']);
    seq.ground_truth = ground_truth;        % 保存groundtruth
    
    seq.init_rect = ground_truth(1,:);      % 初始化数据为groundtruth第一行，[x y w h]    
    target_sz = [ground_truth(1,4), ground_truth(1,3)];
    seq.target_sz = target_sz;
	seq.pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
    
    img_path = seq.video_path;
    img_files_struct = dir(fullfile(img_path, '*.jpg'));
    img_files = {img_files_struct.name};                      % 将所有图片名称保存为cell数组\
    seq.img_files = img_files;
    seq.s_frames = img_files(1, seq.st_frame : seq.en_frame); % 取出configSeq里面设置的有效帧区间
    for i = 1 : length(seq.s_frames)
        seq.s_frames{i} = [img_path seq.s_frames{i}];         % 每一帧都具有完整的路径
    end
    