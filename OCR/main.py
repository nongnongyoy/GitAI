import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('data/1.jpeg'), lang='chi_sim')
text = text.split()
print(text)
