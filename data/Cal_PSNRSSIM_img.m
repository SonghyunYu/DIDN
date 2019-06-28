%% calcluate SSIM 
clear all; clc;

%% image path
folder1 = 'original\'; % original image path
folder2 = 's50\'; % denoised image path


%% calculate SSIM
lists1 = dir(folder1); 
lists2 = dir(folder2); 
n = length(lists1);

ps = zeros(n-2,1);
ss = zeros(n-2,1);
for i=3:n
    file_name1 = strcat(folder1,'\',lists1(i).name);
    a = imread(file_name1);
    file_name2 = strcat(folder2,'\',lists2(i).name);
    b = imread(file_name2);
  
    [ps(i-2), ss(i-2)] = Cal_PSNRSSIM(a, b, 0, 0);
    
end
ps;
ss;
mean_ps = mean(ps)
mean_ss = mean(ss)
