from rembg import remove
from PIL import Image

input_path = 'image.png'
input = Image.open(input_path)
output = remove(input)

output.show()
   