P = imread("lab\lab1\imgs\book.jpg");
imshow(P);

[X, Y] = ginput(4);

for i = 1:4
    fprintf("Point %d : (%f %f)\n",i, X(i), Y(i))
end

% Desired A4 corners in top-left,top-right,bottom-right, bottom-left
x = [0 210 210 0];
y = [0 0 297 297];

A = [];
v = [];


for i = 1:4
    Xi = X(i); Yi = Y(i);   % clicked points in image
    xi = x(i); yi = y(i);   % desired A4 coordinates

    A = [A;
        Xi Yi 1 0 0 0 -xi*Xi -xi*Yi
        0 0 0 Xi Yi 1 -yi*Xi -yi*Yi
    ];
    v = [v; xi; yi];
end

disp("A"); disp(round(A));
disp("v"); disp(v);

% Compute u = A inverse v
u = A \ v;

U = reshape([u;1], 3,3)';
disp("U"); disp(U);

w = U * [X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
disp("Transformed coordinates"); disp(w);

T = maketform('projective', U');
P2 = imtransform(P, T, "xData", [0 210], 'YData', [0 297]);

imshow(P2);

% Convert to appropriate color space
P2_hsv = rgb2hsv(P2);

imshow(P2_hsv);

% Threshold the hue channel for pink color
pink_mask = (P2_hsv(:,:,1) > 0.7 | P2_hsv(:,:,1) < 0.1) & ...
            (P2_hsv(:,:,2) > 0.4) & ...
            (P2_hsv(:,:,3) > 0.6);


pink_mask = imopen(pink_mask, strel('rectangle', [5, 5]));
pink_mask = imclose(pink_mask, strel('rectangle', [10, 10]));

stats = regionprops(pink_mask, 'BoundingBox');

% Display the original transformed image
figure;
imshow(P2);
hold on;

for i = 1:length(stats)
    bbox = stats(i).BoundingBox;
    rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;


