import json
from pathlib import Path


class Config:

    main_dir: Path = Path()
    data_dir: Path = Path('dataset/')
    out_dir: Path = Path('out/')

    _save_path = Path('config.json')

    _cfg = None

    @classmethod
    def get(cls) -> 'Config':
        cfg = cls()
        if Config._cfg is not None:
            return Config._cfg
        if Config._save_path.exists():
            try:
                with open(Config._save_path, 'r') as f:
                    cfg_json = json.load(f)
                cfg.main_dir = Path(cfg_json.get('main_dir', cfg.main_dir))
                cfg.data_dir = Path(cfg_json.get('data_dir', cfg.data_dir))
                cfg.output_dir = Path(cfg_json.get('out_dir', cfg.out_dir))
            except json.JSONDecodeError:
                pass
        cfg.save()
        print('Config loaded.')

        Config._cfg = cfg
        return cfg

    def save(self) -> None:
        if not Config._save_path.exists():
            print('Config saved.')
            with open(Config._save_path, 'w+') as f:
                json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        return {
            'main_dir': str(self.main_dir),
            'data_dir': str(self.data_dir),
            'out_dir': str(self.out_dir)
        }
