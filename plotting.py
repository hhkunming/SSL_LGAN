import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Plot image examples.
def plot_img(img, title):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

def plot_img2(img):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()

def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img

def img_tile_variance(img_original, delta_imgs, n_delta, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0, stretch=False):

    if stretch:
      img_original = img_stretch(img_original)
      delta_imgs = img_stretch(delta_imgs)
    img_original = np.array(img_original)
    delta_imgs = np.array(delta_imgs)
    if img_original.ndim != 3 and delta_imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = delta_imgs.shape[0]

    img_shape = np.array(delta_imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(n_imgs/n_delta * aspect_ratio))
        tile_width = int(np.ceil((1+n_delta) / aspect_ratio))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    tile_img_shape = np.array(delta_imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color

    yoff = (img_shape[0] + border) * grid_shape[0]/2
    tile_img[yoff:yoff+img_shape[0], 0:img_shape[1], ...] = img_original

    for i in range(grid_shape[0]):
        for j in range(1, grid_shape[1]):
            #img = delta_imgs[(j-1) + i*(grid_shape[1]-1)]
            img = delta_imgs[(j-1)*grid_shape[0] + i]
            
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img

def img_tile_tangents(imgs,recon_imgs, tan_imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0, stretch=False):

    if stretch:
      imgs = img_stretch(imgs)
      recon_imgs = img_stretch(recon_imgs)
      tan_imgs = img_stretch(tan_imgs)
    imgs = np.array(imgs)
    recon_imgs = np.array(recon_imgs)
    tan_imgs = np.array(tan_imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]
    n_tan = tan_imgs.shape[1]

    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(n_imgs * aspect_ratio))
        tile_width = int(np.ceil((2+n_tan) / aspect_ratio))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if j == 0:
                img = imgs[i]
            elif j == 1:
                img = recon_imgs[i]
            else:
                img = tan_imgs[i, j-2, ...]

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img

def img_tile_tangents2(imgs, tan_imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0, stretch=False):

    if stretch:
      imgs = img_stretch(imgs)
      tan_imgs = img_stretch(tan_imgs)
    imgs = np.array(imgs)
    tan_imgs = np.array(tan_imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]
    n_tan = tan_imgs.shape[1]

    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(n_imgs * aspect_ratio))
        tile_width = int(np.ceil((1+n_tan) / aspect_ratio))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if j == 0:
                img = imgs[i]
            else:
                img = tan_imgs[i, j-1, ...]

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img

def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0, stretch=False):
    ''' Tile images in a grid.
    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.
    '''

    # Prepare images
    if stretch:
        imgs = img_stretch(imgs)
    imgs = np.array(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i*grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

    return tile_img

def conv_filter_tile(filters):
    n_filters, n_channels, height, width = filters.shape
    tile_shape = None
    if n_channels == 3:
        # Interpret 3 color channels as RGB
        filters = np.transpose(filters, (0, 2, 3, 1))
    else:
        # Organize tile such that each row corresponds to a filter and the
        # columns are the filter channels
        tile_shape = (n_channels, n_filters)
        filters = np.transpose(filters, (1, 0, 2, 3))
        filters = np.resize(filters, (n_filters*n_channels, height, width))
    filters = img_stretch(filters)
    return img_tile(filters, tile_shape=tile_shape)
    
def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.
  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).
  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.
  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image
  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)
  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats
  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not
  :returns: array suitable for viewing as an image.
  (See:`PIL.Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.
  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in range(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in range(tile_shape[0]):
          for tile_col in range(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array
