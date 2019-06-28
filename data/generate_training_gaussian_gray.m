%% Generate training data file for color image denoising
clear; close all; clc;

%% Train Set
folder_origin = 'Train/DIV2K_samples/original';
savepath = 'training_Gray_5to50_uint8_samples.h5';

patch_size = 64;
stride = 64;

data = zeros(patch_size, patch_size, 10000, 'uint8');
label = zeros(patch_size, patch_size, 10000, 'uint8');

count = 0;
list_origin = dir(folder_origin);
n = length(list_origin);


%% generate data
for i = 3 : n
    file_origin = strcat(folder_origin, '\', list_origin(i).name);
    
    origin = imread(file_origin); 
    if size(origin,3) == 3
        origin = rgb2gray(origin);
    end
    [hei, wid] = size(origin);
    
    for x = 1  : stride : hei-patch_size+1
        for y = 1  :stride : wid-patch_size+1
            noiseSigma = randi([1, 10])*5; %5-50
            count=count+1;
            subim_origin = origin(x : x+patch_size-1, y : y+patch_size-1);
            subim_noisy = uint8(single(subim_origin) + noiseSigma*randn(size(subim_origin)));
            
            label(:, :, count) = subim_origin;
            data(:, :, count) = subim_noisy;
        end
    end
    
    display(100*(i-2)/(n-2));display('percent complete');
end

data = data(:,:,1:count);
label = label(:,:,1:count);

order = randperm(count);
data = data(:, :, order);
label = label(:, :, order); 


%% writing to HDF5 (Train)
chunksz = 16;
created_flag = false;
totalct = 0;

for batchno = 1:floor((count)/chunksz)
    batchno;
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,totalct+1], 'lab', [1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz, 'uint8'); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
