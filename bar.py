#from pyzbar.pyzbar import decode
#from PIL import Image
#decode(Image.open('/home/user_reorder04/Desktop/1.jpg'))


from PIL import Image, ImageDraw

from pyzbar.pyzbar import decode


image = Image.open('1.jpg').convert('RGB')
draw = ImageDraw.Draw(image)
for barcode in decode(image):
    rect = barcode.rect
    draw.rectangle(
        (
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top + rect.height)
        ),
        outline='#000000'
    )

    draw.polygon(barcode.polygon, outline='#000000')


image.save('2.png')

