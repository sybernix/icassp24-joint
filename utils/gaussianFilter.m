function gaussian_filter = gaussianFilter(kernel_size, sigma)
    if mod(kernel_size, 2) == 0
        kernel_halfwidth = kernel_size / 2;
    else
        kernel_halfwidth = (kernel_size - 1) / 2;
    end
    
    [Y,X] = meshgrid(-kernel_halfwidth:kernel_halfwidth, -kernel_halfwidth:kernel_halfwidth);
    filter = exp(-(X.^2 + Y.^2)/(2 * sigma^2));
    gaussian_filter = filter / sum(sum(filter));
end