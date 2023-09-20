function psi = preparePsi(filter, patch_height, patch_width, kernel_halfwidth)
    psi = zeros(patch_height * patch_width);
    
    for row = 1 : patch_height
        for col = 1 : patch_width
            temp_patch = zeros(patch_height + 2*kernel_halfwidth, patch_width+ 2*kernel_halfwidth);
            temp_patch(row: row + 2 * kernel_halfwidth, col:col + 2*kernel_halfwidth) = filter;
            psi((row-1)*patch_width + col,:) = reshape(temp_patch(kernel_halfwidth + 1: patch_height + kernel_halfwidth, kernel_halfwidth + 1:patch_width+kernel_halfwidth).',1,[]);
        end
    end
    psi = psi + 1e-8;
    psi = sinkhornKnopp(psi);
end