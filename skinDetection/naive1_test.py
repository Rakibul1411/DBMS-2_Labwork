import numpy as np
import imageio
import matplotlib.pyplot as plt

input_img = "image8.jpeg"
output_img = "detected_skin.jpeg"
fp = open('threshold.txt', "r")   

trained_value = np.zeros(shape=(256, 256, 256))
new_img = imageio.v2.imread(input_img)


for i in range(256):
    for j in range(256):
        for k in range(256):
            val = fp.readline()
            trained_value[i][j][k] = float(val)

# Image dimensions
height, width, _ = new_img.shape
T = 0.4 


for x in range(height):
    for y in range(width):
        red = new_img[x, y, 0]
        green = new_img[x, y, 1]
        blue = new_img[x, y, 2]

    
        if trained_value[red, green, blue] <= T:
            
            new_img[x, y, 0] = 255  
            new_img[x, y, 1] = 0   
            new_img[x, y, 2] = 0   


# Save the result
plt.imsave(output_img, new_img)
print("Done Skin Detection...")
