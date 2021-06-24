class hparams:

    train_or_test = 'train'
    output_dir = 'logs'
    aug = True
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 5000000
    epochs_per_checkpoint = 10
    batch_size = 16
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 256,256,256 # if 2D: 256,256,1
    patch_size = 128,128,128 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.mhd'


    source_train_dir = 'img'
    label_train_dir = 'label'
    source_test_dir = 'img'
    label_test_dir = 'label'

    output_dir_test = 'results'