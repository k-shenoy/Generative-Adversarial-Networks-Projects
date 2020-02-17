from PIL import Image
import os

path = "prelim-data/"
i = 0
for filename in os.listdir(path):
    img = Image.open(path + filename)
    img_resize = img.resize((64, 64), Image.LANCZOS)
    img_resize.save("data/trafficsigns" + str(i) + ".png")
    i = i + 1
