class DumpConfig:
    def __init__(self, enabled=False, out_dir=''):
        self.enabled = enabled
        self.out_dir = out_dir
        self.stage_count = 0


DUMP = DumpConfig()
