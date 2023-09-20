function psi = getBFPsi(patch, spatial_kernel, p_h, p_w, w, sigma_r)
    %
    padded_patch = padarray(patch,[w w],0,'both');
    select_patch = padarray(ones(p_h, p_w),[w w],0,'both');
    
    psi = zeros(p_h * p_w);
    
    i = 1;
    j = 1;
    
    for i=1:p_h
        for j=1:p_w
            window = padded_patch(i:i+2*w, j:j+2*w);
    
            r_kernel = exp(-((window-window(w+1,w+1)).^2)/(2*sigma_r^2));
            bf = spatial_kernel.*r_kernel;
    
            bf_full = zeros(p_h+w, p_w+w);
            bf_full(i:i+2*w,j:j+2*w) = bf;
    %         bf_eff = bf_full(w+1:w+p_h,w+1:w+p_w);
            
            psi((i-1)*p_w + j,:) = reshape(bf_full(w+1:w+p_h,w+1:w+p_w).',1,[]);
        end
    end
    psi = sinkhornKnopp(psi, 'maxiter', 100);
end

