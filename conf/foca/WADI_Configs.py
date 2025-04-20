class Config(object):
    def __init__(self):
        self.lora_rank=16
        self.lora_alpha=16
        self.lora_dropout=0.5
        
        self.lora_tune_epoch = 0
        self.train_epochs = 50 
        self.freeze_length_epoch = 10
        self.change_center_epoch = 10

        self.lora_lr = 1e-6
        self.final_layer_lr = 1e-6
        self.weight_decay = 1e-4
        self.early_stopping = False
        self.hidden_size = 64
        self.clip_grad = None
        self.beta1 = 0.9
        self.beta2 = 0.99

        # datasets
        self.dataset = 'WADI'
        self.window_size = 16
        self.time_step = 16
        self.drop_last = False
        self.batch_size = 512
        self.input_channels = 127


        # Anomaly Detection parameters
        self.nu = 0.001
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify model objective ("one-class" or "soft-boundary")
        self.objective = 'one-class'

        self.center_eps = 0.1
        self.omega1 = 1
        self.omega2 = 1.0

        # units
        self.task_name = "anomaly_detection"
        self.dropout = 0.45
        self.stride = 16
        self.prompt_num = 10
        self.patch_len = 16
        self.d_model = 32
        self.n_heads = 8
        self.e_layers = 3
