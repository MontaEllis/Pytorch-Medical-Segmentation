class hparams:

    train_or_test = 'train'
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 256
    patch_size = 128

    fold_arch = '*.mhd'


    source_train_dir = 'img'
    label_train_dir = 'label'
    source_test_dir = 'img'
    label_test_dir = 'label'

    output_dir_test = 'results'