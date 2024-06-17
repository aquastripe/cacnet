import os


class ConfigV2:

    def __init__(self):
        self.KUPCP_dir = '/workspace/dataset/KU_PCP'

        self.image_size = (224, 224)
        self.data_augmentation = True
        self.keep_aspect_ratio = False

        self.backbone = 'vgg16'
        # training
        self.gpu_id = 1
        self.num_workers = 4
        self.com_batch_size = 64

        prefix = 'composition_classification'
        exp_root = os.path.join(os.getcwd(), 'experiments/CompositionClassify')
        self.exp_name = prefix
        exp_path = os.path.join(exp_root, prefix)
        while os.path.exists(exp_path):
            index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
            try:
                index = int(index) + 1
            except:
                index = 1
            exp_name = prefix + ('_repeat{}'.format(index))
            exp_path = os.path.join(exp_root, exp_name)
        self.exp_path = exp_path

        print('Experiment name {} \n'.format(os.path.basename(exp_path)))
        self.checkpoint_dir = os.path.join(exp_path, 'checkpoints')
        self.log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)


class Config:
    KUPCP_dir = '/workspace/dataset/KU_PCP'

    image_size = (224, 224)
    data_augmentation = True
    keep_aspect_ratio = False

    backbone = 'vgg16'
    # training
    gpu_id = 1
    num_workers = 4
    com_batch_size = 64

    prefix = 'composition_classification'
    exp_root = os.path.join(os.getcwd(), 'experiments/CompositionClassify')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)
    print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)


cfg = ConfigV2()

if __name__ == '__main__':
    cfg = Config()
