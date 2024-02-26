import imgaug.augmenters as iaa
import imgaug.augmenters.imgcorruptlike as icorr
import imgaug
import imageio
import dtlpy as dl
import logging
import tempfile
import os

logger = logging.getLogger('imgaug')


class ServiceRunner(dl.BaseServiceRunner):
    # Dictionary mapping corruption types to their respective ImgAug functions
    image_corrupt_like = {
        'gaussian_noise': icorr.GaussianNoise,
        'shot_noise': icorr.ShotNoise,
        'impulse_noise': icorr.ImpulseNoise,
        'speckle_noise': icorr.SpeckleNoise,
        # 'gaussian_blur': icorr.GaussianBlur,
        # 'glass_blur': icorr.GlassBlur,
        'defocus_blur': icorr.DefocusBlur,
        'motion_blur': icorr.MotionBlur,
        'zoom_blur': icorr.ZoomBlur,
        'fog': icorr.Fog,
        'frost': icorr.Frost,
        'snow': icorr.Snow,
        'spatter': icorr.Spatter,
        'contrast': icorr.Contrast,
        'brightness': icorr.Brightness,
        'saturate': icorr.Saturate,
        'jpeg_compression': icorr.JpegCompression,
        'pixelate': icorr.Pixelate,
        'elastic_transform': icorr.ElasticTransform
    }
    # Dictionary mapping pooling types to ImgAug functions
    poolings = {"max": iaa.MaxPooling,
                "min": iaa.MinPooling,
                "average": iaa.AveragePooling,
                "median": iaa.MedianPooling
                }

    def __init__(self):
        pass

    def load_image(self, item: dl.Item):
        """
        Loads an image from a given dataset item.
        The image is downloaded and read into an array.

        :param item: dl.Item object representing the dataset item to be loaded.
        :return: Numpy array representing the loaded image.
        """
        buffer = item.download(save_locally=False)
        image = imageio.v2.imread(buffer)
        return image

    def save_image(self, image, item: dl.Item, local_path):
        """
        Saves an image to a temporary directory and uploads it to the dataset.
        The method generates a temporary directory, saves the image, and then uploads it.

        :param image: Numpy array of the image to be saved.
        :param item: dl.Item object representing the dataset item associated with the image.
        :param local_path: Local path where the image is to be saved temporarily.
        :return: Uploaded dataset item corresponding to the augmented image.
        """

        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, local_path)
        imageio.imsave(local_path, image)
        dataset = dl.datasets.get(dataset_id=item.dataset.id)
        aug_item = dataset.items.upload(local_path=local_path, remote_path=item.dir)
        return aug_item

    def flip_image(self, item: dl.Item, context: dl.Context):
        """
        Applies a flip augmentation to the image.
        The method determines the type of flip (horizontal or vertical) based on the context,
        applies the flip, and then saves and returns the flipped image.

        :param item: dl.Item object representing the dataset item to be augmented.
        :param context: dl.Context object containing metadata and configurations.
        :return: Dataset item corresponding to the flipped image.
        """

        logger = logging.getLogger('imgaug.flip_image')
        logger.info('Running service Flip')
        image = self.load_image(item)

        node = context.node
        flip_type = node.metadata['customNodeConfig']['flip_type']
        logger.info('Flip type: {}'.format(flip_type))

        if flip_type == 'horizontal':
            flipper = iaa.Fliplr(1.0)
        elif flip_type == 'vertical':
            flipper = iaa.Flipud(1.0)
        else:
            raise ValueError('Invalid flip type')

        flipped_image = flipper.augment_image(image)

        local_path = 'flipped_' + flip_type + '_' + item.name
        flip_item = self.save_image(flipped_image, item, local_path)

        return flip_item

    def change_temperature(self, item: dl.Item, context: dl.Context):
        """
        Alters the color temperature of an image.
        This method adjusts the color temperature of the image based on the temperature value
        specified in the context. The altered image is then saved and returned.

        :param item: dl.Item object representing the dataset item to be augmented.
        :param context: dl.Context object containing metadata and configurations, including the temperature value.
        :return: Dataset item corresponding to the temperature-adjusted image.
        """
        temperature_logger = logging.getLogger('imgaug.change_temperature')
        temperature_logger.info('Running service Temperature')
        image = self.load_image(item)

        node = context.node
        temperature = node.metadata['customNodeConfig']['temperature']
        temperature_logger.info('Temparature: {}'.format(temperature))

        aug = iaa.ChangeColorTemperature((temperature))

        aug_image = aug.augment_image(image)

        local_path = 'temperature_' + str(temperature) + '_' + item.name
        temperature_item = self.save_image(aug_image, item, local_path)

        return temperature_item

    def corrupt_like_image(self, item: dl.Item, context: dl.Context):
        """
        Applies a corruption effect similar to common image distortions.
        Based on the specified corruption type and severity in the context, this method
        corrupts the image and then saves and returns the corrupted image.

        :param item: dl.Item object representing the dataset item to be corrupted.
        :param context: dl.Context object containing metadata and configurations, including the corruption type and severity.
        :return: Dataset item corresponding to the corrupted image.
        """

        corrupt_logger = logging.getLogger('imgaug.corrupt_like_image')
        corrupt_logger.info('Running service Corrupt Like')
        image = self.load_image(item)

        node = context.node
        corrupt_type = node.metadata['customNodeConfig']['corrupt_type']
        severity = node.metadata['customNodeConfig']['severity']
        corrupt_logger.info('Corrupt type: {}'.format(corrupt_type))

        aug = self.image_corrupt_like[corrupt_type]

        aug_image = aug(severity=severity).augment_image(image)

        local_path = 'corrupt_like_' + corrupt_type + '_' + str(severity) + '_' + item.name
        aug_item = self.save_image(aug_image, item, local_path)

        return aug_item

    def pooling(self, item: dl.Item, context: dl.Context):
        """
        Applies a pooling operation to the image.
        This method performs pooling (max, min, average, or median) on the image based on the
        pooling type and kernel size specified in the context. The pooled image is then saved and returned.

        :param item: dl.Item object representing the dataset item to be processed.
        :param context: dl.Context object containing metadata and configurations, including the pooling type and kernel size.
        :return: Dataset item corresponding to the pooled image.
        """

        pooling_logger = logging.getLogger('imgaug.pooling')
        pooling_logger.info('Running service Pooling')

        node = context.node
        pooling_logger.info('Node: {}'.format(node))
        h_value = node.metadata['customNodeConfig']['h_value']
        w_value = node.metadata['customNodeConfig']['w_value']
        pooling_logger.info('Kernel size: {}x{}'.format(h_value, w_value))
        pooling_logger.info('Variables type: {}{}'.format(type(h_value), type(w_value)))

        image = self.load_image(item)

        node = context.node
        pooling_type = node.metadata['customNodeConfig']['pooling_type']
        pooling_logger.info('Pooling type: {}'.format(pooling_type))

        aug = self.poolings[pooling_type](h_value, w_value)

        aug_image = aug.augment_image(image)

        local_path = 'pooling_' + pooling_type + '_' + str(h_value) + 'x' + str(w_value) + '_' + item.name
        aug_item = self.save_image(aug_image, item, local_path)

        return aug_item
