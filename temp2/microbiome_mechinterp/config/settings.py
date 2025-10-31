from dataclasses import dataclass
import yaml

@dataclass
class Config:
    """Configuration for microbiome analysis pipeline"""
    metag_biom: str
    metat_biom: str
    metadata: str
    lineages: str = None
    latent_dim: int = 50
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    sparse_multiplier: int = 3
    l1_reg: float = 0.001
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
