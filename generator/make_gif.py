import glob
import imageio

anim_file = 'learning.gif'

with imageio.get_writer(anim_file, mode='I', loop=0) as writer:
  filenames = glob.glob('training_images/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
