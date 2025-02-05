import numpy as np
import cv2


skin_rgb_cnt = np.empty(shape=(256, 256, 256))
skin_rgb_cnt.fill(0)


non_skin_rgb_cnt = np.empty(shape=(256, 256, 256))
non_skin_rgb_cnt.fill(0)


total_skin_color = 0
total_non_skin_color = 0
total_images = 555
indices = ["%04d" % x for x in range(1000)]


for index in range(total_images):
    mask_image = cv2.imread("Mask/" + str(indices[index]) + ".bmp")
    actual_image = cv2.imread("ibtd/" + str(indices[index]) + ".jpg")
    height, width, channel = mask_image.shape

    for x in range(height):
        for y in range(width):
            mask_blue = mask_image[x, y, 0]
            mask_green = mask_image[x, y, 1]
            mask_red = mask_image[x, y, 2]

            blue = actual_image[x, y, 0]
            green = actual_image[x, y, 1]
            red = actual_image[x, y, 2]

            if mask_blue > 250 and mask_green > 250 and mask_red > 250:
                non_skin_rgb_cnt[red, green, blue] += 1
                total_non_skin_color += 1
            else:
                skin_rgb_cnt[red, green, blue] += 1
                total_skin_color += 1

    print(f"Using Image {index} - Training DONE!")


P_S = total_skin_color / (total_skin_color + total_non_skin_color)  
P_NS = total_non_skin_color / (total_skin_color + total_non_skin_color)


fp = open('threshold.txt', 'w')

for r in range(256):
    for g in range(256):
        for b in range(256):
            # Calculate the likelihoods P(C|S) and P(C|NS)
            P_C_S = skin_rgb_cnt[r, g, b] / total_skin_color  # P(C|S)
            P_C_NS = non_skin_rgb_cnt[r, g, b] / total_non_skin_color  # P(C|NS)
            
            # Calculate the posterior probabilities (without P(C) term)
            posterior_skin = P_C_S * P_S
            posterior_non_skin = P_C_NS * P_NS

            # Use the ratio of posterior probabilities
            if P_C_NS != 0:  # To avoid division by zero
                threshold = posterior_skin / posterior_non_skin
            else:
                threshold = 0.0

            fp.write(f"{threshold}\n")

fp.close()
