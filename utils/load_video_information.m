function seq = load_video_information(type) % OTB100, UAV123_10fps, UAV123, UAV123_20L

switch type
    case 'OTB100'
        video_base_path = 'D:\Res\tracking_data\OTB100';
        video_name = choose_video_OTB(video_base_path);
        seq = load_video_info_OTB([video_base_path '\' video_name]);
        seq.video_name = video_name;
        seq.st_frame = 1;
        seq.en_frame = seq.len;
    case 'UAV123_10fps'
        database_folder = 'D:\Res\tracking_data\UAV123_10fps\data_seq\UAV123_10fps';
        ground_truth_folder = 'D:\Res\tracking_data\UAV123_10fps\anno\UAV123_10fps';
        video_name = choose_video_UAV(ground_truth_folder);
        seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type);
        seq.video_name = video_name;
    case 'UAV123'
        database_folder = 'D:\Res\tracking_data\UAV123\data_seq\UAV123';
        ground_truth_folder = 'D:\Res\tracking_data\UAV123\anno\UAV123';
        video_name = choose_video_UAV(ground_truth_folder);
        seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type);
        seq.video_name = video_name;
    case 'UAV123_20L'
        database_folder = 'D:\Res\tracking_data\UAV123\data_seq\UAV123';
        ground_truth_folder = 'D:\Res\tracking_data\UAV123\anno\UAV20L';
        video_name = choose_video_UAV(ground_truth_folder);
        seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type);
        seq.video_name = video_name;
end