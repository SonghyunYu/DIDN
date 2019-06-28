%% Create gray test data (.mat)
clear;close all;

%% path settings
folder = 'Test/BSD68/original/';  % path to original images
folder_origin = 'Test/BSD68/original_mat_int/';  % path to save original.mat 
folder_noisy = 'Test/BSD68/noisy_mat_s10_int/';  % path to save noisy.mat

%% generate data for VDSR
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'));  dir(fullfile(folder, '*.bmp'));  dir(fullfile(folder, '*.tif'))];
noiseSigma = 10;
    

for i = 1 : length(filepaths)
    origin = imread(fullfile(folder,filepaths(i).name));
    origin = modcrop(origin,8);
    if size(origin,3)==3
        origin = rgb2gray(origin);
    end
    noisy = single(origin) + noiseSigma*randn(size(origin));
    noisy = uint8(noisy);
    
    last = length(filepaths(i).name)-4;
    filename_origin = fullfile(folder_origin, sprintf('%s.mat',filepaths(i).name(1 : last)));
    filename_noisy = fullfile(folder_noisy, sprintf('%s.mat',filepaths(i).name(1 : last)));
    save(filename_origin, 'origin');
    save(filename_noisy, 'noisy');
    
end


