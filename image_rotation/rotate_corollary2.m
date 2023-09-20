clc;
clear;
close all;

noise_mean = 0;
noise_variance = 0.04;
sigma = 0.4;
mu = 0.4;
gamma = 0.5;
kappa = 0.3;
angle_degrees = 20;

img = imread("peppers_gray.png");
img = im2double(img);
% add noise to the image
noisy_img = imnoise(img, 'gaussian', noise_mean, noise_variance);

h = size(noisy_img, 1);
w = size(noisy_img, 2);

% Calculate padding required
max_disp = ceil(sqrt(size(noisy_img,1)^2 + size(noisy_img,2)^2) - max(size(noisy_img)));
img_padded = padarray(noisy_img, [max_disp max_disp]);

% Update maps for the padded image
[mapx, mapy] = getXYMaps(img_padded, angle_degrees);

mapx = mapx(max_disp + 1 : max_disp + h, max_disp + 1 : max_disp + w);
mapy = mapy(max_disp + 1 : max_disp + h, max_disp + 1 : max_disp + w);

p_h = 10;
p_w = 10;

seq_out = zeros(h, w);
graph_output = zeros(h, w);

% generate gaussian kernel for psi
kernel_halfwidth = 1;
kernel_size = 2 * kernel_halfwidth + 1;
g = gaussianFilter(kernel_size, sigma);

for row = 1:h/p_h
    % find the vertical coordinates of the patch
    psv = (row - 1) * p_h + 1;
    pev = row * p_h;
    for col = 1:w/p_w
        % find the end coordinates of the patch
        psh = (col - 1) * p_w + 1;
        peh = col * p_w;

        p_mapx = mapx(psv: pev, psh: peh);
        p_mapy = mapy(psv: pev, psh: peh);

        x_min = floor(min(p_mapx(:)));
        y_min = floor(min(p_mapy(:)));

        x_max = ceil(max(p_mapx(:)));
        y_max = ceil(max(p_mapy(:)));

        img_patch = img_padded(y_min:y_max, x_min:x_max);
        ip_h = size(img_patch, 1);
        ip_w = size(img_patch, 2);

        % construct theta
        theta = zeros(p_h*p_w, ip_h*ip_w);

        for i=1:p_h
            for j=1:p_w
                theta_r = (i-1)*p_w + j;
                
                x = p_mapx(j,i) + 1 - x_min;
                y = p_mapy(j,i) + 1 - y_min;
        
                l = floor(x);
                t = floor(y);
        
                a = x - l;
                b = y - t;
        
                theta_lt_loc = (t-1) * ip_w + l;
                theta_rt_loc = theta_lt_loc + 1;
        
                theta_lb_loc = t*ip_w + l;
                theta_rb_loc = theta_lb_loc + 1;
        
                theta(theta_r, theta_lt_loc) = (1-b)*(1-a);
                theta(theta_r, theta_rt_loc) = (1-b)*a;
                theta(theta_r, theta_lb_loc) = b*(1-a);
                theta(theta_r, theta_rb_loc) = b*a;
            end
        end
        y = reshape(img_patch.',[],1);

        M = size(theta, 2);
        N = M;

        psi_bar = preparePsi(g, p_h, p_w, kernel_halfwidth);

        output_flat = psi_bar*theta*y;
        output_mat = reshape(output_flat, p_h, p_w);
        seq_out(psv: pev, psh: peh) = output_mat;

        % calculate graph output
        H = zeros(M, M + N);
        for k=1:M
            H(k,k) = 1;
        end

        G = zeros(N,M+N);
        for k=1:N
            G(k,M+k) = 1;
        end

        theta_square = rand(M, M);
        theta_square = theta_square ./ sum(theta_square, 2); % make theta row stochastic
        theta_square(1:size(theta, 1), :) = theta;

        psi_bar_full = zeros(M, M);
        psi_bar_full(1:p_h*p_w, 1:p_h*p_w) = psi_bar;
        rans_psi_comp = rand(M-p_h*p_w, M-p_h*p_w);
        rans_psi_comp = rans_psi_comp * rans_psi_comp';
        psi_bar_full(p_h*p_w+1:end, p_h*p_w+1:end) = rans_psi_comp;

        psi_bar_full = sinkhornKnopp(psi_bar_full);

%         eigenVs = eig(psi_bar_full);
% 
%         % Check if the eigenvalues are real
%         if all(isreal(eigenVs))
%             disp('All eigenvalues of psi are real');
%         else
%             disp('Not all eigenvalues of psi are real');
%         end
%         
%         if max(eigenVs) <= (1 + 1e-8) && min(eigenVs) >0
%             disp('Eigenvalues of psi are within (0, 1]');
%         else
%             disp('Eigenvalues of psi are not within (0, 1]');
%         end

        Lbar = (inv(psi_bar_full) - eye(M)) / mu;

        Amn = inv(theta_square);
        A = zeros(M + N, M + N);
        A(1:M, M+1:M+N) = Amn;

        I = eye(M+N);
        C = H' * H + gamma * (I-A)'*H'*H*(I-A) + kappa*(G'*Lbar*G);

        x = pcg(C, H'* y, 1e-6, 3000);
        diff = y - x(1:M);
        output_mat_graph = reshape(x(M+1:M+p_h*p_w), p_h, p_w);
        graph_output(psv: pev, psh: peh) = output_mat_graph;

        a=1;
    end
end

figure,
imshow(graph_output);

groundtruth = imrotate(img, angle_degrees, 'bilinear', 'crop');

graph_psnr = psnr(graph_output, groundtruth);
seq_psnr = psnr(seq_out, groundtruth);

psnr_gain = graph_psnr - seq_psnr

% calculate SSIM
graph_ssim = ssim(graph_output, groundtruth);
seq_ssim = ssim(seq_out, groundtruth);

ssim_gain = graph_ssim - seq_ssim


function [x_rot, y_rot] = getXYMaps(originalImage, angle_degrees)
    [rows, cols, ~] = size(originalImage);
    alpha = -deg2rad(angle_degrees); % Negative because the y-axis is reversed in image coordinates
    
    R = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)]; % Rotation matrix
    
    % Create a grid for the original image
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    % Apply the inverse rotation to the grid
    coords = R \ [X(:) - cols/2, Y(:) - rows/2]'; 
    x_rot = reshape(coords(1, :) + cols/2, rows, cols);
    y_rot = reshape(coords(2, :) + rows/2, rows, cols);
end




