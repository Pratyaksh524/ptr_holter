from PIL import Image, ImageDraw

SIZE = 1024
img = Image.new("RGBA", (SIZE, SIZE), (18, 35, 62, 255))
d = ImageDraw.Draw(img)

# Soft blue rounded app tile.
d.rounded_rectangle((28, 28, 996, 996), radius=190, fill=(24, 67, 105, 255), outline=(104, 211, 221, 255), width=18)

# Holter monitor body.
d.rounded_rectangle((250, 155, 774, 875), radius=90, fill=(240, 247, 250, 255), outline=(18, 35, 62, 255), width=24)
d.rounded_rectangle((315, 220, 709, 470), radius=34, fill=(15, 39, 61, 255), outline=(91, 207, 214, 255), width=12)

# Screen glow / status marks.
d.rounded_rectangle((365, 275, 659, 400), radius=18, fill=(26, 88, 105, 255))
d.ellipse((390, 315, 418, 343), fill=(114, 239, 186, 255))
d.rounded_rectangle((445, 316, 607, 342), radius=12, fill=(114, 239, 186, 255))

# ECG trace across the monitor display.
trace = [(374, 365), (418, 365), (440, 365), (460, 330), (482, 395), (505, 300), (527, 365), (570, 365), (594, 345), (615, 365), (652, 365)]
d.line(trace, fill=(255, 93, 104, 255), width=14, joint="curve")

# Holter controls and indicator.
d.ellipse((358, 540, 430, 612), fill=(91, 207, 214, 255), outline=(18, 35, 62, 255), width=10)
d.ellipse((474, 540, 546, 612), fill=(255, 220, 112, 255), outline=(18, 35, 62, 255), width=10)
d.rounded_rectangle((590, 550, 674, 602), radius=20, fill=(18, 35, 62, 255))

# Cable and electrode accent.
d.line([(512, 875), (512, 914), (650, 914), (708, 850)], fill=(240, 247, 250, 255), width=22, joint="curve")
d.ellipse((690, 818, 760, 888), fill=(255, 93, 104, 255), outline=(240, 247, 250, 255), width=12)

img.save("holter_monitor_icon.png")
img.save("holter_monitor_icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
