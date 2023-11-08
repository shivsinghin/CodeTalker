def get_model(cfg):
    if cfg.arch == 'vertices_encoder':
        from models.vertices_encoder import VQAutoEncoder as Model
        model = Model(args=cfg)
    elif cfg.arch == 'audio2vertices':
        from models.audio2vertices import Audio2Vertices as Model
        model = Model(args=cfg)    
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model