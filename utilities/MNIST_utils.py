
try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy as np
import os
import functools
import operator
import struct
import array

def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse

    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file

    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


#%% Visualizations.


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

    This function==useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row==a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i]==None:
                # if channel==None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

#%%



#%% For MNIST


def gen_monte_carlo_movie(PGM, Lchains = 100, Nchains = 10, Nthermalize = 0, N_PT = 10, Nstep = 1,PT_type = None ,p_init = 0.5, mean = False,exp= '0', folder = 'monte_carlo_videos', avconv = 'avconv'):
    size = int(np.sqrt(PGM.n_visibles))
    size2 = size**2
    if PT_type==None:
        PT_type = PGM.PT_type
    datav, datah = PGM.gen_data(record_replica = True, Nthermalize = Nthermalize, Nchains = Nchains, Lchains = Lchains, N_PT = N_PT, Nstep = Nstep,PT_type = PT_type,beta_min = 0, p_init = p_init);
    if mean:
        if N_PT > 1:
            for i in range(Nchains):
                datav[i,:,:,:] = PGM._mean_visibles_block(datah[i,:,:,:],PT =PT_type)
        else:
            for i in range(Nchains):
                datav[i,:,:] = PGM.mean_visibles(datah[i,:,:])

    for i in range(Lchains):
        if N_PT >1:
            image = Image.fromarray(
            tile_raster_images(
            X= datav[:,i,:,:].reshape([Nchains*N_PT,size2]),
            img_shape=(size, size),
            tile_shape=(Nchains, N_PT),
            tile_spacing=(2, 2)
        )
        );
        else:
            image = Image.fromarray(
            tile_raster_images(
            X= datav[:,i,:].reshape([Nchains,size2]),
            img_shape=(size, size),
            tile_shape=(Nchains, 1),
            tile_spacing=(2, 2)
        )
        );


        image.save(folder+ '/' + exp+ 'tmp_%08d.png'%i)
    os.system(avconv + ' -y -i ' + folder + '/'+  exp + 'tmp_%08d.png ' + folder + '/' + 'monte_carlo_p%s_%s.mp4'%(p_init,exp))
    os.system('rm ' + folder + '/' + exp + '*tmp*')

def gen_FP_movie(PGM,fantasy_particles, exp = '0', folder = 'fantasy_particles_videos',avconv = 'avconv'):
    size = int(np.sqrt(PGM.n_visibles))
    size2 = size**2
    for i in range(len(fantasy_particles)):
        m = np.minimum(PGM.nchains,20)
        if PGM.N_PT >1:
            image = Image.fromarray(
            tile_raster_images(
            X= fantasy_particles[i][:m,:].reshape([m*PGM.N_PT,size2]),
            img_shape=(size, size),
            tile_shape=(m, PGM.N_PT),
            tile_spacing=(2, 2)
        )
        );
        else:
            image = Image.fromarray(
            tile_raster_images(
            X= fantasy_particles[i][:m,:].reshape([m,size2]),
            img_shape=(size, size),
            tile_shape=(m, 1),
            tile_spacing=(2, 2)
        )
        );
        image.save(folder + '/' + exp + 'tmp_%08d.png'%i)
    os.system(avconv + ' -y -i ' + folder + '/'+  exp + 'tmp_%08d.png ' + folder + '/' + 'fantasy_particles_%s.mp4'%exp)
    os.system('rm ' + folder + '/' + exp + '*tmp*')


def gen_W_movie(PGM,all_weights,exp='0',folder = 'weights_videos',avconv = 'avconv'):
    size = int(np.sqrt(PGM.n_visibles))
    a,b = np.percentile(PGM.weights.flatten(),[1,99.9]);
    for i in range(len(all_weights)):
        weights= (all_weights[i]- a)/(b-a) * 255
        weights = np.array(np.minimum(weights,255),dtype='uint8')

        image = Image.fromarray(
        tile_raster_images(
            X= weights,
            img_shape=(size, size),
            tile_shape=( int(PGM.n_hiddens/10)+1,10),
            tile_spacing=(2, 2),scale_rows_to_unit_interval=False,output_pixel_vals = False
        )
        )

        image.save(folder + '/' + exp + 'tmp_%08d.png'%i)
    os.system(avconv + ' -y -i ' + folder + '/'+  exp + 'tmp_%08d.png ' + folder + '/' + 'weigths_%s.mp4'%exp)
    os.system('rm ' + folder + '/' + exp + '*tmp*')


def show_weights(PGM,sort='false', show = True, n_h = None, columns = None, rows = None):
    if n_h is None:
        n_h = PGM.n_hiddens
    if columns is None:
        columns = 10
    if rows is None:
        rows = int(PGM.n_hiddens/10)+1
    size = int(np.sqrt(PGM.n_visibles))
    if sort == 'false':
        weights = PGM.weights[:n_h]
    elif sort =='beta':
        a = np.argsort( (PGM.weights**2).sum(1)  )
        weights = PGM.weights[a[::-1],:][:n_h]
    elif sort == 'c':
        a = np.argsort( PGM.c_plus + PGM.c_minus  )
        weights = PGM.weights[a[::-1],:][:n_h]

    image = Image.fromarray(
    tile_raster_images(
        X= weights,
        img_shape=(size, size),
        tile_shape=(rows ,columns),
        tile_spacing=(2, 2)
    )
    )
    if show:
        image.show()
    return image




def show_FP(PGM, show = True):
    size = int(np.sqrt(PGM.n_visibles))
    if hasattr(PGM,'fantasy_h'):
        if PGM.N_PT>1:
            datah = PGM.fantasy_h[:,:,0]
        else:
            datah = PGM.fantasy_h[:,:]
        datav = PGM.mean_visibles(datah)
    else:
        if PGM.N_PT>1:
            datav = PGM.fantasy_v[:,:,0]
        else:
            datav = PGM.fantasy_v[:,:]

    image = Image.fromarray(
    tile_raster_images(
        X= datav,
        img_shape=(size, size),
        tile_shape=( 10,PGM.nchains/10),
        tile_spacing=(2, 2)
    )
    )
    if show:
        image.show()
    return image


def show_samples(data,size=None,tile_shape=None,show=True):
    data = np.asarray(data,dtype=float)
    if size is None:
        side = int(np.sqrt(data.shape[1]))
        size = (side,side)
    if tile_shape is None:
        side_viz = int(np.sqrt(data.shape[0]))
        tile_shape = (side_viz,side_viz)
    image = Image.fromarray(
    tile_raster_images(
        X= data,
        img_shape=size,
        tile_shape= tile_shape,
        tile_spacing=(2, 2)
    )
    )
    if show:
        image.show()
    return image


#%%
