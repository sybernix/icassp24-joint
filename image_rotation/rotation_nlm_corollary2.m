clc;
clear;
close all;

noise_mean = 0;
noise_variance = 0.01;
mu = 0.3;
gamma = 0.6;
kappa = 0.2;
angle_degrees = 20;
condThreshold = 500;

p = 3; % nlm patch size
s = 9; % search window size

% Filter parameter for NLM
h_filter = 1;
% gauss_norm = h_filter^2 * p^2;
gauss_norm = 0.4;

% Define patch and window bounds
patch_radius = (p-1)/2;
window_radius = (s-1)/2;

img = imread("lena_gray.png");
img = im2double(img);
% add noise to the image
noisy_img = imnoise(img, 'gaussian', noise_mean, noise_variance);

h = size(noisy_img, 1);
w = size(noisy_img, 2);

% Calculate padding required
max_disp = ceil(sqrt(size(noisy_img,1)^2 + size(noisy_img,2)^2) - max(size(noisy_img)));
img_padded = padarray(noisy_img, [max_disp max_disp]);

% this is used for constructing NLM psi
nlm_psi_padded_img = padarray(noisy_img, [h w], 'symmetric');
nlm_psi_rotated_img = imrotate(nlm_psi_padded_img, angle_degrees, 'bilinear', 'crop');
nlm_psi_rotated_img = nlm_psi_rotated_img(h+1:2*h,w+1:2*w);

% Update maps for the padded image
[mapx, mapy] = getXYMaps(img_padded, angle_degrees);

mapx = mapx(max_disp + 1 : max_disp + h, max_disp + 1 : max_disp + w);
mapy = mapy(max_disp + 1 : max_disp + h, max_disp + 1 : max_disp + w);

p_h = 10;
p_w = 10;

seq_out = zeros(h, w);
graph_output = zeros(h, w);

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

        interpolated_patch = nlm_psi_rotated_img(psv: pev, psh: peh);

        M = size(theta, 2);
        psi_bar = getNLMPsi(window_radius, p_h, p_w, interpolated_patch, patch_radius, p, gauss_norm);

        eigenVs = eig(psi_bar);
        min_eigval = min(eigenVs);
        if min_eigval < 0
            delta = abs(min_eigval) + 1e-5;  % small constant for safety
            psi_bar = psi_bar + delta * eye(size(psi_bar));
            psi_bar = sinkhornKnopp(psi_bar);
            eigenVs = eig(psi_bar);
        end

        N = M;

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

        % Check if the matrix is invertible
%         if rank(theta_square) == N
%             disp('Theta is invertible');
%         else
%             disp('Theta is not invertible');
%             is_theta_ok = false;
%         end
%         
%         % Check if the condition number exceeds the threshold
%         if cond(theta_square) > condThreshold
%             disp('Theta is ill-c/onditioned/ close to singular');
%             is_theta_ok = false;
%         else
%             disp('Theta conditioning is OK');
%         end

        psi_bar_full = zeros(M, M);
        psi_bar_full(1:p_h*p_w, 1:p_h*p_w) = psi_bar;
        rans_psi_comp = rand(M-p_h*p_w, M-p_h*p_w);
        rans_psi_comp = rans_psi_comp * rans_psi_comp';
        psi_bar_full(p_h*p_w+1:end, p_h*p_w+1:end) = rans_psi_comp;

        psi_bar_full = sinkhornKnopp(psi_bar_full);

        output_flat = psi_bar_full(1:p_h*p_w, 1:p_h*p_w)*theta*y;
        output_mat = reshape(output_flat, p_h, p_w);
        seq_out(psv: pev, psh: peh) = output_mat;

        Lbar = (inv(psi_bar_full) - eye(M)) / mu;

        Amn = inv(theta_square);
        A = zeros(M + N, M + N);
        A(1:M, M+1:M+N) = Amn;

        I = eye(M+N);
        C = H' * H + gamma * (I-A)'*H'*H*(I-A) + kappa*(G'*Lbar*G);

        x = pcg(C, H'* y, 1e-6, 3000);
%         x = inv(C)*H'*y;
%         x = linsolve(C, H'*y);
        diff = y - x(1:M);
        output_mat_graph = reshape(x(M+1:M+p_h*p_w), p_h, p_w);
        graph_output(psv: pev, psh: peh) = output_mat_graph;

        a=1;
    end
    row
end

figure,
imshow(graph_output);
figure,
imshow(seq_out);

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

function psi_bar = getNLMPsi(window_radius, p_h, p_w, interpolated_patch, patch_radius, p, gauss_norm)
    % Initialize the Psi matrix for the current patch
    psi_bar = zeros(p_h*p_w, p_h*p_w);
    
    for x = 1:p_h
        for y = 1:p_w
            % Define bounds for the current window around (x, y)
            iMinW = max(x - window_radius, 1);
            iMaxW = min(x + window_radius, p_h);
            jMinW = max(y - window_radius, 1);
            jMaxW = min(y + window_radius, p_w);
    
            % Define bounds for the NLM patch centered at (x, y)
            iMinP = max(x - patch_radius, 1);
            iMaxP = min(x + patch_radius, p_h);
            jMinP = max(y - patch_radius, 1);
            jMaxP = min(y + patch_radius, p_w);
    
            % Extract the patch centered at (x, y)
            patch_center = interpolated_patch(iMinP:iMaxP, jMinP:jMaxP);
    
            % Index for the current pixel in Psi
            idx = sub2ind([p_h, p_w], x, y);

            % Adjust the range for xx and yy to stay within (a, b) patch
            xx_min = max(iMinW, 1);
            xx_max = min(iMaxW, p_h);
            yy_min = max(jMinW, 1);
            yy_max = min(jMaxW, p_w);
    
            for xx = xx_min:xx_max
                for yy = yy_min:yy_max
                    % Define bounds for the NLM patch centered at (xx, yy)
                    iiMinP = max(xx - patch_radius, 1);
                    iiMaxP = min(xx + patch_radius, p_h);
                    jjMinP = max(yy - patch_radius, 1);
                    jjMaxP = min(yy + patch_radius, p_w);
    
                    % Extract the patch centered at (xx, yy)
                    patch = interpolated_patch(iiMinP:iiMaxP, jjMinP:jjMaxP);
    
                    % Resize patches if necessary (to handle boundary patches)
                    if size(patch,1) ~= p || size(patch,2) ~= p
                        patch = imresize(patch, [p p]);
                    end
                    if size(patch_center,1) ~= p || size(patch_center,2) ~= p
                        patch_center = imresize(patch_center, [p p]);
                    end

                    % Compute the weight
                    weight = exp(-sum((patch(:) - patch_center(:)).^2) / gauss_norm);
    
%                     idx_neighbor = sub2ind([p_h, p_w], xx-row+1, yy-col+1);
                            
                    % Accumulate the weight in the Psi matrix
                    idx_neighbor = sub2ind([p_h, p_w], xx, yy);
                    psi_bar(idx, idx_neighbor) = weight;
                end
            end
            
            % Normalize the row in Psi for current pixel
            psi_bar(idx, :) = psi_bar(idx, :) / sum(psi_bar(idx, :));
        end
    end
end




