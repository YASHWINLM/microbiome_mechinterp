import argparse
from .data_processing.processor import MicrobiomeDataProcessor
from .models.vae import VAE
from .training.trainer import MicrobiomeVAETrainer
from .feature_engineering.extractor import FeatureExtractor
from .visualization.visualizer import Visualizer
from .utils.helpers import get_device

def main():
    parser = argparse.ArgumentParser(description='Microbiome Mechinterp Pipeline')
    parser.add_argument('--metag-biom', required=True, help='MetaG BIOM file')
    parser.add_argument('--metat-biom', required=True, help='MetaT BIOM file')
    parser.add_argument('--metadata', required=True, help='Metadata file')
    parser.add_argument('--lineages', help='Lineages file')
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    device = get_device()
    
    data_paths = {
        'metag_biom': args.metag_biom,
        'metat_biom': args.metat_biom,
        'metadata': args.metadata
    }
    if args.lineages:
        data_paths['lineages'] = args.lineages
    
    # Load and process data
    processor = MicrobiomeDataProcessor(data_paths, device=device)
    processor.load_data()
    processor.apply_rclr_transformation()
    processor.filter_features()
    processor.prepare_datasets()
    
    # Train VAE
    vae = VAE(input_dim=processor.metag_scaled.shape[1], latent_dim=args.latent_dim).to(device)
    trainer = MicrobiomeVAETrainer(vae, device=device)
    train_losses, _ = trainer.train(processor.metag_scaled, epochs=args.epochs)
    
    # Extract features
    latent = trainer.get_latent_representation(processor.metag_scaled)
    extractor = FeatureExtractor(vae, device)
    extractor.train_sparse_autoencoder(latent)
    sparse_features = extractor.get_sparse_features(latent)
    
    # Visualize
    viz = Visualizer()
    viz.visualize_latent_space(latent, processor.encoded_labels, processor.label_names, "MetaG")
    viz.visualize_sparse_features(sparse_features, processor.encoded_labels, processor.label_names, "MetaG")
    
    print("Pipeline complete!")

if __name__ == '__main__':
    main()
